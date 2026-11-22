import torch
import torch.nn.functional as F

from octave import Cache, selected_tokens


def rms(x, weight=None, eps=1e-5):
    y = x.float()
    result = (y * (y.square().mean(-1, keepdim=True) + eps).rsqrt()).to(x.dtype)
    return result if weight is None else result * weight


def rotate(x, pos, theta):
    width = 64
    f = torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width
    phase = pos[:, None].float() / theta**f
    pairs = x[..., -width:].reshape(len(x), -1, width // 2, 2)
    c, s = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    a, b = pairs.float().unbind(-1)
    rotated = torch.stack((a*c - b*s, b*c + a*s), -1).to(x.dtype)
    return torch.cat((x[..., :-width], rotated.reshape_as(x[..., -width:])), -1)


def collapse(x, fn, scale, base):
    flat = x.flatten(1).float()
    mix = F.linear(flat, fn.float()) * (flat.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    gates = (mix.reshape(len(x), -1, 4) * scale.float().reshape(1, -1, 1) + base.float().reshape(1, -1, 4)).sigmoid()
    pre = gates[:, 0] + 1e-6
    merged = (pre[..., None] * x).sum(1).to(x.dtype)
    return merged, (gates[:, 1] * 2 + 1e-6) if gates.size(1) == 2 else None


class DecoderLayer:
    def __init__(self, weights, layer, capacity):
        weights.require_layer(layer)
        self.w = weights
        self.layer = layer
        self.prefix = f"model.layers.{layer}."
        self.sliding = weights.config["layer_types"][layer] == "sliding_attention"
        self.cache = Cache(capacity, self.sliding, weights.device)
        self.theta = 10000 if self.sliding else 10000000

    def raw(self, name):
        return self.w.raw(self.prefix + name)

    def linear(self, name, x):
        return self.w.linear(self.prefix + name, x)

    def engram(self, x, history, ids):
        p = "engram_layer."
        e = self.w.lookup(self.layer, history, ids)
        def head_norm(value, weight=None):
            return F.rms_norm(value, (value.size(-1),), weight)
        norms = torch.stack([self.raw(p + f"lookup_per_head_norms.{i}.weight") for i in range(16)])
        e = head_norm(e).mul(norms).flatten(1)
        v = self.linear(p + "value_proj", e) + self.raw(p + "value_proj.bias")
        v = head_norm(v) * self.raw(p + "unified_pre_gamma")
        streams = []
        for i in range(4):
            name = p + f"key_projs.{i}"
            k = self.linear(name, e) + self.raw(name + ".bias")
            k = head_norm(k, self.raw(p + f"norm_key.{i}.weight"))
            q = head_norm(x[:, i], self.raw(p + f"norm_query.{i}.weight"))
            dot = (k * q).sum(-1, keepdim=True) * 3072**-0.5
            gate = (dot.abs().clamp_min(1e-6).sqrt() * dot.sign()).sigmoid()
            streams.append(gate * v + self.raw(p + f"unified_per_hc_scales.{i}") * v)
        active = torch.tensor([t not in (120002, 120120, 120124) for t in ids], device=x.device)
        return x + torch.stack(streams, 1).masked_fill(~active[:, None, None], 0)

    def index(self, x, qres, positions):
        p = "self_attn.compressed_indexer."
        k = rms(self.linear(p + "token_key_projection", x), self.raw(p + "token_key_norm.weight"))
        q = rotate(self.linear(p + "query_projection", qres).reshape(-1, 16, 128), positions, self.theta)
        head = self.linear(p + "head_weight_projection", x).float() * (16 * 128)**-0.5
        cache = self.cache
        result = []
        for i in range(len(x)):
            pos = cache.end + i
            cache.pending[pos % 12] = k[i]
            if pos % 4 == 3:
                group = cache.pending[pos % 12 - 3:pos % 12 + 1].float()
                gates = self.linear(p + "key_gate_projection", group).float() + self.raw(p + "intra_group_ape")
                pooled = (gates.softmax(0) * self.linear(p + "key_value_projection", group)).sum(0, keepdim=True)
                key = rms(pooled, self.raw(p + "key_norm.weight")).to(k.dtype)
                cache.groups[pos // 4] = rotate(key, positions[i:i+1] - 3, self.theta)[0]
            scores = q.new_empty(0)
            if pos >= 2048:
                dots = (q[i].float() @ cache.groups[:pos//4].float().T).relu()
                scores = head[i] @ dots
            result.append(selected_tokens(scores, pos))
        return result

    def attention(self, x):
        p = "self_attn."
        n = len(x)
        cache = self.cache
        pos = torch.arange(cache.end, cache.end+n, device=x.device)
        qres = rms(self.linear(p+"q_a_proj", x), self.raw(p+"q_a_layernorm.weight"))
        q = rotate(self.linear(p+"q_b_proj", qres).reshape(n,-1,256), pos, self.theta)
        kv = self.linear(p+"kv_a_proj_with_mqa", x)
        latent = rms(kv[:, :512], self.raw(p+"kv_a_layernorm.weight"))
        rotary = rotate(kv[:, 512:], pos, self.theta)
        selected = self.index(x, qres, pos) if not self.sliding else [
            torch.arange(max(0, cache.end+i-511), cache.end+i+1, device=x.device) for i in range(n)]
        start = cache.end
        history = torch.unique(torch.cat(selected), sorted=True)
        past = history[history < start]
        old_latent, old_rotary = cache.get(past) if len(past) else (latent[:0], rotary[:0])
        for i in range(n):
            cache.append(latent[i], rotary[i])
        recent_latent, recent_rotary = cache.get(pos)
        all_latent = torch.cat((old_latent, recent_latent))
        all_rope = torch.cat((old_rotary, recent_rotary))
        expanded = self.linear(p+"kv_b_proj", all_latent).reshape(len(all_latent), q.size(1), 448)
        keys, values = expanded.split((192,256), -1)
        attention = []
        for i, indices in enumerate(selected):
            idx = torch.searchsorted(torch.cat((past, pos)), indices)
            dots = torch.einsum("hd,thd->ht", q[i,:,:192].float(), keys[idx].float())
            dots += q[i,:,192:].float() @ all_rope[idx].float().T
            scores = (dots.to(q.dtype) * 0.0625).float()
            sinks = self.raw(p+"learnable_sink_param").reshape(-1, 1)
            prob = torch.cat((scores, sinks.float()), -1).softmax(-1)[:,:-1].to(x.dtype)
            attention.append(torch.einsum("ht,thd->hd", prob, values[idx]))
        self.last_selected = selected
        a = torch.stack(attention).reshape(n, -1) * self.linear(p+"linear_gate", x).sigmoid()
        return rms(self.linear(p+"o_proj", a), self.raw(p+"post_proj_norm.weight"))

    def mlp(self, x):
        def swiglu(prefix, values, clip=False):
            gate = self.linear(prefix + ".gate_proj", values)
            up = self.linear(prefix + ".up_proj", values)
            if clip:
                gate, up = gate.clamp_max(10), up.clamp(-10, 10)
            return self.linear(prefix + ".down_proj", F.silu(gate) * up)
        if self.layer == 0:
            return rms(swiglu("mlp", x), self.raw("mlp.post_fc2_norm.weight"))
        scores = F.linear(x.float(), self.raw("mlp.gate.weight").float()).sigmoid()
        ids = (scores + self.raw("mlp.gate.e_score_correction_bias")).topk(16).indices
        prob = scores.gather(1, ids)
        prob = prob / (prob.sum(-1, keepdim=True) + 1e-20) * 3.998
        latent = self.linear("mlp.routed_expert_down_proj", x)
        combined = torch.zeros_like(latent)
        host_ids = ids.cpu()
        for rank, device in enumerate(self.w.devices):
            local = [e for e in host_ids.unique().tolist()
                     if self.w.owners is None or self.w.owners[self.layer-1][e] == rank]
            if not local:
                continue
            values, weights = latent.to(device), prob.to(device)
            out = torch.zeros_like(values)
            for expert in local:
                rows, slots = torch.where(host_ids == expert)
                rows, slots = rows.to(device), slots.to(device)
                v = swiglu(f"mlp.experts.{expert}", values[rows], True)
                out.index_add_(0, rows, (v * weights[rows, slots, None]).to(out.dtype))
            combined += out.to(x.device)
        y = self.linear("mlp.routed_expert_up_proj", combined) + swiglu("mlp.shared_experts", x)
        return rms(y, self.raw("mlp.combine_output_norm.weight"))

    def __call__(self, x, history, ids):
        if self.layer+1 in self.w.config["engram_layer_ids"]:
            x = self.engram(x, history, ids)
        for prefix, norm, sublayer in (("hc_attn_layer", "input_layernorm", self.attention),
                                      ("hc_mlp_layer", "post_attention_layernorm", self.mlp)):
            p = prefix + ".hc_pre.hc_"
            merged, post = collapse(x, self.raw(p+"fn"), self.raw(p+"scale"), self.raw(p+"base"))
            value = sublayer(rms(merged, self.raw(norm+".weight")))
            x = (post[..., None] * value[:,None].float() + x.float()).to(x.dtype)
        return x
