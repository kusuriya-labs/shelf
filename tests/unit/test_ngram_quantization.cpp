#include <shelf/quantization/ngram/ngram.h>
#include <shelf/io/safetensors_writer.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>
#include <unistd.h>
using namespace shelf::quantization::ngram;
namespace fs = std::filesystem;
int passed = 0, failed = 0;
void check(bool v, const char* what) { if (v) ++passed; else { ++failed; std::printf("FAIL: %s\n", what); } }
uint16_t half(float v) { return std::bit_cast<uint16_t>(static_cast<_Float16>(v)); }
Table table(unsigned bits, const char* id) { Table t; t.id=id; t.name=id; t.shard="data.safetensors"; t.codes_tensor=t.id+".codes"; t.scale_tensor=t.id+".scale"; t.codebook_tensor=t.id+".codebook"; t.ngram_order=2; t.rows=2; t.dim=9; t.bits=bits; return t; }
void write_table(const fs::path& p, const Table& t, std::vector<uint8_t> codes, std::vector<uint16_t> scales, std::vector<uint16_t> cb, bool wrong_shape=false, bool wrong_dtype=false, bool missing=false) {
    auto w = shelf::io::SafetensorsWriter::create(p.string()); check(w.ok(),"writer create"); if (!w.ok()) return;
    auto& x=w.value(); auto width=row_bytes(t.dim,t.bits).value();
    check(x.plan_tensor(t.codes_tensor,wrong_dtype ? shelf::io::SType::I8 : shelf::io::SType::U8,{t.rows,wrong_shape ? width-1 : width}).ok(),"plan codes");
    check(x.plan_tensor(t.scale_tensor,shelf::io::SType::F16,{t.rows}).ok(),"plan scale");
    if (!missing) check(x.plan_tensor(t.codebook_tensor,shelf::io::SType::F16,{std::size_t(1u<<t.bits)}).ok(),"plan codebook");
    check(x.write_header().ok(),"header");
    if (wrong_shape) codes.resize(t.rows*(width-1));
    check(x.write_tensor(t.codes_tensor,codes.data(),codes.size()).ok(),"write codes");
    check(x.write_tensor(t.scale_tensor,scales.data(),scales.size()*2).ok(),"write scales");
    if (!missing) check(x.write_tensor(t.codebook_tensor,cb.data(),cb.size()*2).ok(),"write codebook");
    check(x.finish().ok(),"finish");
}
int main() {
    std::mt19937 rng(42); std::normal_distribution<float> normal;
    std::vector<float> samples(100000); for (auto& x:samples) x=normal(rng);
    std::vector<uint8_t> known4{1,2,3}, packed4(2); check(pack(known4,4,packed4).ok() && packed4==std::vector<uint8_t>({0x21,0x03}),"4 bit order");
    std::vector<uint8_t> known5{0,1,2,3,4,5,6,7}, packed5(5); check(pack(known5,5,packed5).ok() && packed5==std::vector<uint8_t>({0x20,0x88,0x41,0x8a,0x39}),"5 bit order");
    fs::path root=fs::temp_directory_path() / ("shelf-ngram-"+std::to_string(getpid())); fs::create_directories(root);
    for (unsigned bits : {4u,5u}) {
        auto trained=train(samples,bits); check(trained.ok(),"train"); if (!trained.ok()) continue; auto cb=trained.value();
        for (std::size_t dim : {1u,2u,7u,8u,9u,15u,16u,96u,160u,256u}) {
            std::vector<uint8_t> idx(dim), back(dim), packed(row_bytes(dim,bits).value()); for (auto& v:idx) v=rng()%(1u<<bits);
            check(pack(idx,bits,packed).ok() && unpack(packed,bits,back).ok() && idx==back,"pack round trip");
            std::vector<float> zero(dim), decoded(dim,1); uint16_t scale=1;
            check(quantize(zero,cb,bits,scale,packed).ok() && scale==0 && std::all_of(packed.begin(),packed.end(),[](auto v){return v==0;}) && decode(packed,scale,cb,bits,decoded).ok() && std::all_of(decoded.begin(),decoded.end(),[](float v){return v==0 && !std::signbit(v);}),"zero row");
            if (dim>=96) {
                std::vector<float> row(dim); for (auto& x:row) x=normal(rng); check(quantize(row,cb,bits,scale,packed).ok(),"quantize");
                std::vector<uint16_t> bf(dim); check(decode_bf16(packed,scale,cb,bits,bf).ok() && decode(packed,scale,cb,bits,decoded).ok(),"bf16 decode");
                bool bf_ok=true; for (std::size_t j=0;j<dim;++j) { auto u=std::bit_cast<uint32_t>(decoded[j]); auto expected=uint16_t((u+0x7fff+((u>>16)&1))>>16); bf_ok &= bf[j]==expected; } check(bf_ok,"bf16 rounding");
                auto start=std::chrono::steady_clock::now(); for(int i=0;i<10000;++i) decode(packed,scale,cb,bits,decoded);
                auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/10000;
                double err=0, ref=0; for (std::size_t i=0;i<dim;++i) { err+=(row[i]-decoded[i])*(row[i]-decoded[i]); ref+=row[i]*row[i]; }
                check(std::sqrt(err/ref)<(bits==4 ? .16 : .10),"quantization accuracy"); std::printf("bits=%u dim=%zu %.1f ns/row\n",bits,dim,ns);
            }
        }
        double err_sum=0, ref_sum=0;
        for (int r=0;r<200;++r) { std::vector<float> row(256), out(256); for (auto& v:row) v=normal(rng); std::vector<uint8_t> c(row_bytes(row.size(),bits).value()); uint16_t scale=0; check(quantize(row,cb,bits,scale,c).ok() && decode(c,scale,cb,bits,out).ok(),"gaussian row"); for (std::size_t i=0;i<row.size();++i) { err_sum+=(row[i]-out[i])*(row[i]-out[i]); ref_sum+=row[i]*row[i]; } }
        double relative=std::sqrt(err_sum/ref_sum); std::printf("bits=%u gaussian relative RMSE=%.4f\n",bits,relative); check(relative < (bits==4 ? 0.11 : 0.065),"gaussian accuracy");
        Table t=table(bits,"a"); Manifest m; m.profile=bits==4?"ngram4":"ngram5"; m.source_model="test"; m.tables={t};
        auto width=row_bytes(t.dim,bits).value(); std::vector<uint8_t> codes(t.rows*width); std::vector<uint16_t> scales{0,half(1)};
        write_table(root/t.shard,t,codes,scales,cb); auto serialized=serialize_manifest(m); check(serialized.ok(),"serialize"); auto parsed=parse_manifest(serialized.value()); check(parsed.ok() && validate_bundle(root,parsed.value()).ok(),"bundle round trip");
        std::ofstream(root / "ngram_manifest.json") << serialized.value();
        std::ifstream manifest_file(root / "ngram_manifest.json"); std::string manifest_text((std::istreambuf_iterator<char>(manifest_file)), {}); check(parse_manifest(manifest_text).ok(), "manifest file round trip");
        auto reader=TableReader::open(root,t); std::vector<float> out(t.dim,1); check(reader.ok() && reader.value().decode_row(0,out).ok() && out[0]==0,"read zero row");
        codes[0]=1; write_table(root/t.shard,t,codes,scales,cb); check(!validate_bundle(root,m).ok(),"zero row violation"); codes[0]=0;
        codes[width*2-1]=128; write_table(root/t.shard,t,codes,scales,cb); check(!validate_bundle(root,m).ok(),"padding violation"); codes[width*2-1]=0;
        scales[1]=0x8000; write_table(root/t.shard,t,codes,scales,cb); check(!validate_bundle(root,m).ok(),"negative zero scale");
        scales[1]=0x7c00; write_table(root/t.shard,t,codes,scales,cb); check(!validate_bundle(root,m).ok(),"infinite scale");
        scales[1]=half(-1); write_table(root/t.shard,t,codes,scales,cb); check(!validate_bundle(root,m).ok(),"negative scale"); scales[1]=half(1);
        auto badcb=cb; badcb[0]=0x7e00; write_table(root/t.shard,t,codes,scales,badcb); check(!validate_bundle(root,m).ok(),"nan codebook");
        badcb=cb; for (auto& x:badcb) x=half(0.5f); write_table(root/t.shard,t,codes,scales,badcb); check(!validate_bundle(root,m).ok(),"unnormalized codebook");
        badcb=cb; badcb[0]=half(1); write_table(root/t.shard,t,codes,scales,badcb); check(!validate_bundle(root,m).ok(),"unordered codebook");
        write_table(root/t.shard,t,codes,scales,cb,true); check(!validate_bundle(root,m).ok(),"wrong shape");
        write_table(root/t.shard,t,codes,scales,cb,false,true); check(!validate_bundle(root,m).ok(),"wrong dtype");
        write_table(root/t.shard,t,codes,scales,cb,false,false,true); check(!validate_bundle(root,m).ok(),"missing tensor");
        m.tables[0].shard="../escape"; check(!validate_bundle(root,m).ok(),"path escape");
        m.tables[0].shard="/etc/passwd"; check(!validate_bundle(root,m).ok(),"absolute path");
        fs::path link=root/"escape-link"; std::error_code ec; fs::create_symlink("/etc/passwd",link,ec); if (!ec) { m.tables[0].shard="escape-link"; check(!validate_bundle(root,m).ok(),"symlink escape"); }
        m.tables[0]=t; m.tables[0].bits=3; check(!validate_manifest(m).ok(),"bad bits");
        check(!parse_manifest("{}").ok(),"missing manifest fields");
        auto good=parse_manifest(serialized.value()); check(good.ok(),"parse valid manifest");
        auto altered=good.value(); altered.format="unknown"; check(!validate_manifest(altered).ok(),"unknown format");
        altered=good.value(); altered.version=2; check(!validate_manifest(altered).ok(),"unknown version");
        altered=good.value(); altered.codec="unknown"; check(!validate_manifest(altered).ok(),"unknown codec");
        altered=good.value(); altered.profile="unknown"; check(!validate_manifest(altered).ok(),"unknown profile");
        altered=good.value(); altered.profile=bits==4?"ngram5":"ngram4"; check(!validate_manifest(altered).ok(),"profile mismatch");
        altered=good.value(); altered.tables[0].rows=0; check(!validate_manifest(altered).ok(),"zero rows");
        altered=good.value(); altered.tables[0].dim=0; check(!validate_manifest(altered).ok(),"zero dim");
        altered=good.value(); altered.tables[0].codes_tensor="wrong"; check(!validate_manifest(altered).ok(),"wrong tensor name");
        altered=good.value(); altered.tables[0].shard="missing"; check(!validate_bundle(root,altered).ok(),"missing shard");
        check(!parse_manifest("{\"format\":true}").ok(),"wrong type");
    }
    Manifest mixed; mixed.profile="mixed"; mixed.source_model="test"; mixed.tables={table(4,"a"),table(5,"b")}; mixed.tables[0].shard="a.safetensors"; mixed.tables[1].shard="b.safetensors";
    for (const auto& t : mixed.tables) { auto cb=train(samples,t.bits).value(); auto width=row_bytes(t.dim,t.bits).value(); write_table(root/t.shard,t,std::vector<uint8_t>(t.rows*width),std::vector<uint16_t>(t.rows),cb); }
    check(validate_manifest(mixed).ok() && parse_manifest(serialize_manifest(mixed).value()).ok() && validate_bundle(root,mixed).ok(),"mixed profile");
    mixed.tables[1].id="a"; check(!validate_manifest(mixed).ok(),"duplicate id");
    fs::remove_all(root); std::printf("test_ngram_quantization: passed=%d failed=%d\n",passed,failed); return failed ? 1 : 0;
}
