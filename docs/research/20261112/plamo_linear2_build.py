import argparse
import json
import shlex
import subprocess
from pathlib import Path
from plamo_linear_build import generated_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fused-input', action='store_true')
    parser.add_argument('--input-rows', type=int, default=1)
    parser.add_argument('--input-slice', action='store_true')
    parser.add_argument('--shuffle-load', action='store_true')
    parser.add_argument('--stream-a', action='store_true')
    parser.add_argument('--fused-output', action='store_true')
    parser.add_argument('--output-rows', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = root / 'build-gfx1201'
    output = build / 'plamo-linear2'
    output.mkdir(exist_ok=True)
    generated = generated_kernel(root)
    anchor = '    if (variant == 1) {'
    generated = generated.replace(anchor, '''    if (variant == 7) {
        if (mt == 1) return launch_variant<B, 1, false, 8>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        if (mt == 2) return launch_variant<B, 2, false, 8>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        if (mt == 4) return launch_variant<B, 4, false, 8>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        return launch_variant<B, 8, false, 8>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    }
    if (variant == 8) return launch_variant<B, 8, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    if (variant == 9) {
        if (mt == 1) return launch_variant<B, 1, false, 4, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        if (mt == 2) return launch_variant<B, 2, false, 4, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        return launch_variant<B, 4, false, 4, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    }
''' + anchor, 1)
    selected = subprocess.check_output(['git','show','3d72e00:src/shelf/kernels/optimized/linear/exl3_family/mul24/projection.hip'],cwd=root,text=True)
    baseline = selected.replace('namespace shelf::kernel {','namespace plamo_baseline {\nusing namespace shelf::kernel;',1)
    if args.stream_a:
        def stream_a(source):
            while '        int32x2_t afrag[MT][8];' in source:
                a=source.index('        int32x2_t afrag[MT][8];')
                b=source.index('        int32x8_t acc[MT];',a)
                source=source[:a]+source[b:]
            source=source.replace('true, afrag[mt][s], true, b,','true, a, true, b,')
            anchor='            for (int mt = 0; mt < MT; ++mt) {\n                acc[mt] ='
            replacement='            for (int mt = 0; mt < MT; ++mt) {\n                int32x2_t a=*reinterpret_cast<const int32x2_t*>(arow[mt]+g*kBlock+s*16u);\n                if(!arow_ok[mt]) {a[0]=0;a[1]=0;}\n                acc[mt] ='
            if anchor not in source:raise ValueError('stream-A WMMA anchor missing')
            return source.replace(anchor,replacement)
        generated=stream_a(generated)
        generated=generated.replace('exl3_mul24_gemm_kernel<B, MT, B == 4,','exl3_mul24_gemm_kernel<B, MT, B == 4 && MT <= 4,')
        selected=stream_a(selected)
    if args.fused_output:
        selected=selected.replace('template <int BITS, int MT, bool PIPE = true>','template <int BITS, int MT, bool PIPE = true, int WAVES = 4, bool FUSED = false>')
        selected=selected.replace('uint32_t tile_x, uint32_t tile_z) {','uint32_t tile_x, uint32_t tile_z, const PlamoExl3Projection* ep = nullptr) {')
        selected=selected.replace('const uint32_t nt = nb * 4u + wave;','const uint32_t nt = nb * WAVES + wave;')
        selected=selected.replace('    yprime += static_cast<size_t>(tile_z) * rows * n;','    if constexpr (!FUSED) yprime += static_cast<size_t>(tile_z) * rows * n;')
        anchor='    for (int mt = 0; mt < MT; ++mt) {\n#pragma unroll\n        for (int i = 0; i < 8; ++i) {\n            const uint32_t m = m0 + mt * 16u + half * 8u + i;\n            if (m < rows)'
        a=selected.index(anchor)
        a-=len('#pragma unroll\n')
        b=selected.index('\n}\n\ntemplate <int B,',a)
        selected=selected[:a]+r"""    __shared__ float epi[MT*16*128];
    for (int mt=0;mt<MT;++mt) {
        for(int i=0;i<8;++i) {
            const uint32_t m=m0+mt*16u+half*8u+i;
            if(m<rows) {
                const float v=accf[mt][i]*::shelf::quantization::exl3::kExl3Int8GridStep;
                if constexpr(FUSED) epi[(m-m0)*128u+wave*16u+col]=v;
                else yprime[static_cast<size_t>(m)*n+n0+col]=v;
            }
        }
    }
    if constexpr(FUSED) {
        __syncthreads();
        for(uint32_t lm=wave;lm<16u*MT && m0+lm<rows;lm+=WAVES) {
            const uint32_t local_col=lane*4u,col=nb*128u+local_col;
            const float4 v=*reinterpret_cast<const float4*>(epi+lm*128u+local_col);
            float v0=0.0f+v.x,v1=0.0f+v.y,v2=0.0f+v.z,v3=0.0f+v.w;
            exl3_detail::fwht128(v0,v1,v2,v3,lane);
            v0=(v0*kInvSqrt128)*f16_bits_to_f32(ep->svh[col]);
            v1=(v1*kInvSqrt128)*f16_bits_to_f32(ep->svh[col+1u]);
            v2=(v2*kInvSqrt128)*f16_bits_to_f32(ep->svh[col+2u]);
            v3=(v3*kInvSqrt128)*f16_bits_to_f32(ep->svh[col+3u]);
            const size_t index=static_cast<size_t>(m0+lm)*ep->out_stride+col;
            if(ep->out_bf16) {
                const uint2 packed={uint32_t(f32_to_bf16_rne_device(v0))|(uint32_t(f32_to_bf16_rne_device(v1))<<16u),
                    uint32_t(f32_to_bf16_rne_device(v2))|(uint32_t(f32_to_bf16_rne_device(v3))<<16u)};
                *reinterpret_cast<uint2*>(reinterpret_cast<uint16_t*>(ep->out)+index)=packed;
            } else *reinterpret_cast<float4*>(reinterpret_cast<float*>(ep->out)+index)={v0,v1,v2,v3};
        }
    }
"""+selected[b:]
        anchor='template <int B, int MT, bool PIPE>\nhipError_t launch_group_bits'
        a=selected.index(anchor)
        fused=r"""
template<int BITS>
__global__ void __launch_bounds__(256) group_gemm_out(PlamoExl3ProjectionGroup args) {
    uint32_t tile=blockIdx.x;
    for(uint32_t i=0u;i<args.count;++i) {
        const auto p=args.p[i];
        if(tile>=p.n/128u) {tile-=p.n/128u;continue;}
        group_body<BITS,1,true,8,true>(p.codes,p.scales,args.k,args.scale_stride_bytes/4u,
            reinterpret_cast<const uint32_t*>(p.trellis),p.yprime,args.rows,nullptr,
            p.n,args.k,args.k/128u,1u,tile,0u,&p);
        return;
    }
}
hipError_t launch_fused_output(const PlamoExl3ProjectionGroup& args,uint32_t bits,hipStream_t stream) {
    uint32_t blocks=0u;for(uint32_t i=0u;i<args.count;++i)blocks+=args.p[i].n/128u;
    if(bits==4u)group_gemm_out<4><<<blocks,256,0,stream>>>(args);
    else if(bits==5u)group_gemm_out<5><<<blocks,256,0,stream>>>(args);
    else group_gemm_out<8><<<blocks,256,0,stream>>>(args);
    return hipGetLastError();
}
"""
        selected=selected[:a]+fused+selected[a:]
        selected=selected.replace('if(ep->out_bf16)', 'if constexpr(true)')
        anchor='    if (bits == 4u)\n        err = args.rows'
        selected=selected.replace(anchor,f'    bool fuse_out=args.rows<={args.output_rows}u;\n    for(uint32_t i=0u;i<args.count;++i)fuse_out=fuse_out && args.p[i].splits==1u;\n    if(fuse_out)return launch_fused_output(args,bits,stream);\n'+anchor,1)
    if args.shuffle_load:
        selected=selected.replace('cur[s][1] = tile[i1];','cur[s][1] = BITS == 4 ? __shfl(cur[s][0], (lane + 1u) & 31u) : tile[i1];')
        selected=selected.replace('nxt[s][1] = tile[i1];','nxt[s][1] = BITS == 4 ? __shfl(nxt[s][0], (lane + 1u) & 31u) : tile[i1];')
        selected=selected.replace('cur[s][2] = kNeedW2 ? tile[i2] : 0u;','cur[s][2] = BITS == 8 ? __shfl(cur[s][0], (lane + 1u) & 31u) : (kNeedW2 ? tile[i2] : 0u);')
        selected=selected.replace('nxt[s][2] = kNeedW2 ? tile[i2] : 0u;','nxt[s][2] = BITS == 8 ? __shfl(nxt[s][0], (lane + 1u) & 31u) : (kNeedW2 ? tile[i2] : 0u);')
    if args.fused_input:
        start = selected.index('    const uint32_t m = blockIdx.x;', selected.index('group_rotate_f4('))
        finish = selected.index('\n}\n__global__', start)
        rotate = selected[start:finish].replace('const uint32_t m = blockIdx.x;', 'const uint32_t m = 0u;').replace('g += 8u', 'g += 4u')
        fused = r"""
template<int BITS>
__global__ void __launch_bounds__(128) group_gemm_fused(PlamoExl3ProjectionGroup args) {
    uint32_t tile = blockIdx.x;
    for (uint32_t i=0u; i<args.count; ++i) {
        const auto p = args.p[i];
        const uint32_t blocks = p.n / 64u;
        if (tile >= blocks * p.splits) { tile -= blocks * p.splits; continue; }
        extern __shared__ uint8_t scratch[];
        auto codes = reinterpret_cast<int8_t*>(scratch);
        auto scales = reinterpret_cast<float*>(scratch + args.k);
        const auto input = args.input;
        const auto suh = p.suh;
        const uint32_t input_row_stride=args.input_stride, code_row_stride=args.k;
        const uint32_t scale_row_stride_floats=args.k/128u, groups=args.k/128u;
""" + rotate + r"""
        __syncthreads();
        group_body<BITS,1,BITS!=5>(codes,scales,args.k,args.k/128u,
            reinterpret_cast<const uint32_t*>(p.trellis),p.yprime,1u,nullptr,
            p.n,args.k,(args.k/128u+p.splits-1u)/p.splits,1u,tile%blocks,tile/blocks);
        return;
    }
}
hipError_t launch_fused_input(const PlamoExl3ProjectionGroup& args,uint32_t bits,hipStream_t stream) {
    uint32_t blocks=0u,n=0u;
    for(uint32_t i=0u;i<args.count;++i) { blocks+=args.p[i].n/64u*args.p[i].splits; n=std::max(n,args.p[i].n); }
    const uint32_t shared=args.k+args.k/128u*4u;
    if(bits==4u) group_gemm_fused<4><<<blocks,128,shared,stream>>>(args);
    else if(bits==5u) group_gemm_fused<5><<<blocks,128,shared,stream>>>(args);
    else group_gemm_fused<8><<<blocks,128,shared,stream>>>(args);
    hipError_t err=hipGetLastError();if(err!=hipSuccess)return err;
    group_out<<<dim3(1u,(n/128u+3u)/4u,args.count),128,0,stream>>>(args);
    return hipGetLastError();
}
"""
        if args.input_rows > 1:
            fused=fused.replace('scratch + args.k','scratch + args.k*args.rows')
            fused=fused.replace('const uint32_t m = 0u;', 'for(uint32_t m=0u;m<args.rows;++m) {')
            fused=fused.replace('        __syncthreads();', '        }\n        __syncthreads();')
            fused=fused.replace('p.yprime,1u,nullptr','p.yprime,args.rows,nullptr')
            fused=fused.replace('const uint32_t shared=args.k+args.k/128u*4u;', 'const uint32_t shared=(args.k+args.k/128u*4u)*args.rows;')
            fused=fused.replace('dim3(1u,','dim3(args.rows,')
        if args.input_slice:
            selected=selected.replace('uint32_t tile_x, uint32_t tile_z) {','uint32_t tile_x, uint32_t tile_z, uint32_t code_group_base=0u) {')
            a=selected.index('template <int BITS, int MT, bool PIPE = true>')
            b=selected.index('template <int B, int MT, bool PIPE>',a)
            body=selected[a:b].replace('arow[mt] + g * kBlock + s * 16u','arow[mt] + (g-code_group_base) * kBlock + s * 16u')
            body=body.replace('scale_row_stride_floats + g]', 'scale_row_stride_floats + g-code_group_base]')
            selected=selected[:a]+body+selected[b:]
            fused=fused.replace('        extern __shared__ uint8_t scratch[];', '        const uint32_t gps=(args.k/128u+p.splits-1u)/p.splits;\n        const uint32_t g0=(tile/blocks)*gps,g1=min(args.k/128u,g0+gps);\n        extern __shared__ uint8_t scratch[];')
            fused=fused.replace('scratch + args.k*args.rows','scratch + gps*128u*args.rows').replace('scratch + args.k','scratch + gps*128u')
            fused=fused.replace('code_row_stride=args.k','code_row_stride=gps*128u').replace('scale_row_stride_floats=args.k/128u','scale_row_stride_floats=gps')
            fused=fused.replace('for (uint32_t g = wave; g < groups; g += 4u)', 'for (uint32_t g = g0+wave; g < g1; g += 4u)')
            fused=fused.replace('code_row_stride + col)', 'code_row_stride + col-g0*128u)')
            fused=fused.replace('scale_row_stride_floats + g]', 'scale_row_stride_floats + g-g0]')
            fused=fused.replace('group_body<BITS,1,BITS!=5>(codes,scales,args.k,args.k/128u,', 'group_body<BITS,1,BITS!=5>(codes,scales,gps*128u,gps,')
            fused=fused.replace('1u,tile%blocks,tile/blocks);', '1u,tile%blocks,tile/blocks,g0);')
            fused=fused.replace('const uint32_t shared=args.k+args.k/128u*4u;', 'uint32_t shared=0u;for(uint32_t i=0u;i<args.count;++i)shared=std::max(shared,(args.k/128u+args.p[i].splits-1u)/args.p[i].splits*132u);')
            fused=fused.replace('const uint32_t shared=(args.k+args.k/128u*4u)*args.rows;', 'uint32_t shared=0u;for(uint32_t i=0u;i<args.count;++i)shared=std::max(shared,(args.k/128u+args.p[i].splits-1u)/args.p[i].splits*132u*args.rows);')
        index=selected.index('template <int B, int MT, bool PIPE>\nhipError_t launch_group_bits')
        selected=selected[:index]+fused+selected[index:]
        anchor='    if (args.rows >= 32u && args.input_stride % 4u == 0u)'
        selected=selected.replace(anchor,f'    if (args.rows <= {args.input_rows}u) return launch_fused_input(args,bits,stream);\n'+anchor,1)
    selected = selected.replace('namespace shelf::kernel {', 'namespace plamo_selected {\nusing namespace shelf::kernel;', 1)
    selected += '''
namespace plamo_selected {
hipError_t selected_stage(int,PlamoExl3ProjectionGroup args,uint32_t bits) {
    return plamo_selected::launch_plamo_exl3_projection(args,bits,nullptr);
}
}
'''
    main_source = (Path(__file__).with_name('plamo_linear_micro.hip')).read_text()
    anchor = '    if(sweep == 4) {'
    extension = '''    if(sweep == 6) {
        for (int v : {0,1,7,8,9}) {
            for(int mt : (v==0 ? std::vector<int>{0} : v==8 ? std::vector<int>{8} : std::vector<int>{1,2,4})) {
                for(uint32_t sp : {base_sp,2u,4u}) {
                    if(sp>k/128u) continue;
                    gemm(0,0,sp);
                    check(hipMemcpy(ref.p,y.p,size_t(m)*n*sp*4,hipMemcpyDeviceToDevice));
                    gemm(v,mt,sp);
                    std::vector<float> before(size_t(m)*n*sp),after(before.size());
                    check(hipMemcpy(before.data(),ref.p,before.size()*4,hipMemcpyDeviceToHost));
                    check(hipMemcpy(after.data(),y.p,after.size()*4,hipMemcpyDeviceToHost));
                    size_t diff=0;for(size_t i=0;i<before.size();++i) if(std::memcmp(&before[i],&after[i],4))++diff;
                    std::fprintf(stderr,"check v=%d mt=%d sp=%u diff=%zu\\n",v,mt,sp,diff);
                    if(diff) return 3;
                    print("gemm",v,mt,sp,true,[&]{gemm(v,mt,sp);});
                }
            }
        }
    } else if(sweep == 4) {'''
    full = r"""    if(sweep == 7) {
        shelf::kernel::PlamoExl3ProjectionGroup args{};
        args.input=reinterpret_cast<const shelf::bf16_t*>(x.p);args.rows=m;args.k=k;args.input_stride=k;args.scale_stride_bytes=ss;args.count=1;
        auto& p=args.p[0];p.trellis=t.p;p.suh=suh.p;p.svh=svh.p;p.codes=a.p;p.scales=s.p;p.yprime=y.p;
        p.out=reinterpret_cast<shelf::bf16_t*>(out.p);p.n=n;p.splits=base_sp;p.out_stride=n;
        auto before=[&]{check(shelf::kernel::launch_plamo_exl3_projection(args,bits,nullptr));};
        auto after=[&]{check(plamo_selected::launch_plamo_exl3_projection(args,bits,nullptr));};
        before();std::vector<uint16_t> ref(size_t(m)*n),candidate(ref.size());
        check(hipMemcpy(ref.data(),out.p,ref.size()*2,hipMemcpyDeviceToHost));after();
        check(hipMemcpy(candidate.data(),out.p,ref.size()*2,hipMemcpyDeviceToHost));
        size_t diff=0;for(size_t i=0;i<ref.size();++i)if(ref[i]!=candidate[i])++diff;
        std::fprintf(stderr,"full diff=%zu\n",diff);if(diff)return 3;
        std::vector<float> a_us,b_us;
        for(int run=0;run<5;++run) {
            if(run%2){b_us.push_back(time(after,true,1)[0]);a_us.push_back(time(before,true,1)[0]);}
            else {a_us.push_back(time(before,true,1)[0]);b_us.push_back(time(after,true,1)[0]);}
        }
        emit("linear",0,0,base_sp,true,a_us);emit("linear",11,0,base_sp,true,b_us);
    } else """
    main_source = main_source.replace(anchor, full + extension, 1)
    source = output / ((args.output.name + '.hip') if args.output else 'micro.hip')
    source.write_text(generated + baseline + selected + main_source.replace('check(shelf::kernel::launch_plamo_exl3_projection(args,bits,nullptr))','check(plamo_baseline::launch_plamo_exl3_projection(args,bits,nullptr))'))
    commands = json.loads((build / 'compile_commands.json').read_text())
    command = shlex.split(next(c['command'] for c in commands if c['file'].endswith('test_exl3_mul24_linear.hip')))
    command = command[:command.index('-o')] + ['-x','hip',str(source),'-o',str(args.output.resolve() if args.output else output/'micro'),'-x','none',
        str(build/'libshelf_kernels_optimized.a'),
        '-L/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib',
        '-L/opt/rocm/core-7.14/lib','-lamdhip64','-lc++','-lc++abi']
    subprocess.run(command,cwd=root,check=True)


if __name__ == '__main__':
    main()
