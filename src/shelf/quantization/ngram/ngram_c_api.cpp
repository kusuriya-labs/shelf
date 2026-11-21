#include <shelf/quantization/ngram/ngram.h>
#include <shelf/quantization/ngram/ngram_c_api.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <fstream>
#include <iterator>
#include <thread>

using namespace shelf::quantization::ngram;
namespace {
int error(const std::string& s, char* out, size_t n) { if (out && n) { std::strncpy(out, s.c_str(), n-1); out[n-1]=0; } return 1; }
template<class F> int guard(char* e, size_t n, F f) { try { return f(); } catch (const std::exception& x) { return error(x.what(), e, n); } catch (...) { return error("unknown error",e,n); } }
}
extern "C" {
int shelf_ngram_train_codebook_f32(const float* x, size_t count, unsigned bits, uint16_t* cb, char* e, size_t n) {
 return guard(e,n,[&] { if (!x || !cb || !count) return error("invalid input",e,n); auto r=train({x,count},bits); if (!r.ok()) return error(r.status().message(),e,n); std::copy(r.value().begin(),r.value().end(),cb); return 0; });
}
int shelf_ngram_quantize_rows_bf16(const uint16_t* x, size_t rows, size_t dim, unsigned bits, const uint16_t* cb, uint8_t* codes, uint16_t* scales, unsigned threads, char* e, size_t n) {
 return guard(e,n,[&] { auto w=row_bytes(dim,bits); if (!w.ok() || !x || !cb || !codes || !scales || !rows || !threads || rows > SIZE_MAX/dim || rows > SIZE_MAX/w.value()) return error("invalid input",e,n);
 std::atomic<bool> failed{false}; std::vector<std::thread> workers; threads=unsigned(std::min<size_t>(threads,rows));
 try { for (unsigned t=0;t<threads;++t) workers.emplace_back([&,t] { try { std::vector<float> row(dim); for (size_t r=rows*t/threads;r<rows*(t+1)/threads && !failed;++r) { for (size_t i=0;i<dim;++i) row[i]=std::bit_cast<float>(uint32_t(x[r*dim+i])<<16); auto st=quantize(row,{cb,size_t(1u<<bits)},bits,scales[r],{codes+r*w.value(),w.value()}); if (!st.ok()) failed=true; } } catch (...) { failed=true; } }); } catch (...) { for (auto& t:workers) t.join(); throw; }
 for (auto& t:workers) t.join(); return failed ? error("row quantization failed",e,n) : 0; });
}
int shelf_ngram_decode_rows_f32(const uint8_t* codes, const uint16_t* scales, const uint16_t* cb, size_t rows, size_t dim, unsigned bits, float* out, char* e, size_t n) {
 return guard(e,n,[&] { auto w=row_bytes(dim,bits); if (!w.ok() || !codes || !scales || !cb || !out || !rows || rows > SIZE_MAX/dim || rows > SIZE_MAX/w.value()) return error("invalid input",e,n); for(size_t r=0;r<rows;++r) { auto st=decode({codes+r*w.value(),w.value()},scales[r],{cb,size_t(1u<<bits)},bits,{out+r*dim,dim}); if(!st.ok()) return error(st.message(),e,n); } return 0; });
}
int shelf_ngram_verify_bundle(const char* root, char* e, size_t n) {
 return guard(e,n,[&] { if (!root) return error("invalid root",e,n); std::ifstream f(std::filesystem::path(root)/"ngram_manifest.json"); if (!f) return error("cannot open manifest",e,n); std::string s((std::istreambuf_iterator<char>(f)),{}); auto m=parse_manifest(s); if (!m.ok()) return error(m.status().message(),e,n); auto st=validate_bundle(root,m.value()); return st.ok()?0:error(st.message(),e,n); });
}
}
