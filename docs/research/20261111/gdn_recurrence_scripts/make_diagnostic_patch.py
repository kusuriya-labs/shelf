#!/usr/bin/env python3
"""Write an opt-in patch for one fixed-position verify-logits row per round."""
from pathlib import Path
import difflib
root=Path(__file__).resolve().parents[4]
name='src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp'
p=root/name
old=p.read_text()
needle='''        BatchExecutionOutput output = executed.release();
        if (output.num_outputs < total_rows) {'''
insert='''        BatchExecutionOutput output = executed.release();
        // Diagnostic only: append selected verify logits and describe their input rows.
        // The caller checks the preceding committed and draft tokens against the fixed prefix.
        if (const char* path = std::getenv("SHELF_GDN_DIAG_LOGITS")) {
            const char* target_string = std::getenv("SHELF_GDN_DIAG_POSITION");
            if (target_string == nullptr || plans.size() != 1u || !plans[0].batched ||
                output.logits.ndim() != 2u || output.logits.element_size() != sizeof(float)) {
                return Status::invalid_argument("invalid GDN diagnostic configuration", __FILE__, __LINE__);
            }
            const uint32_t target = static_cast<uint32_t>(std::strtoul(target_string, nullptr, 10));
            const SlotPlan& plan = plans[0];
            // position is the input token index; this row predicts position+1.
            if (target > plan.position && target <= plan.position + plan.total_rows) {
                const uint32_t row = target - plan.position - 1u;
                const std::size_t vocab = output.logits.dim(1);
                std::vector<float> host(vocab);
                std::vector<int32_t> diag_ids(total_rows);
                if (hipStreamSynchronize(leader.stream) != hipSuccess ||
                    hipMemcpy(host.data(), output.logits.data<float>() + row * vocab,
                              vocab * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
                    return Status::invalid_state("GDN diagnostic logits copy failed", __FILE__, __LINE__);
                }
                if (bridge) {
                    if (hipMemcpy(diag_ids.data(), context.verify_token_ids_device,
                                  total_rows * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) {
                        return Status::invalid_state("GDN diagnostic token copy failed", __FILE__, __LINE__);
                    }
                } else { diag_ids = token_ids; }
                FILE* file = std::fopen(path, "ab");
                if (file == nullptr) return Status::invalid_state("GDN diagnostic open failed", __FILE__, __LINE__);
                const bool ok = std::fwrite(host.data(), sizeof(float), vocab, file) == vocab;
                std::fclose(file);
                if (!ok) return Status::invalid_state("GDN diagnostic write failed", __FILE__, __LINE__);
                std::fprintf(stderr, "GDN_DIAG position=%u target=%u row=%u vocab=%zu ids=",
                             plan.position, target, row, vocab);
                for (uint32_t j = 0; j <= row; ++j)
                    std::fprintf(stderr, "%s%d", j ? "," : "", diag_ids[j]);
                std::fprintf(stderr, "\\n");
            }
        }
        if (output.num_outputs < total_rows) {'''
assert old.count(needle)==1
new=old.replace(needle,insert)
out=Path(__file__).with_name('verify_logits_diagnostic.patch')
out.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name)))
print(out)
