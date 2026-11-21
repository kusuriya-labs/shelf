#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
int shelf_ngram_train_codebook_f32(const float* samples, size_t count, unsigned bits, uint16_t* codebook, char* error, size_t error_size);
int shelf_ngram_quantize_rows_bf16(const uint16_t* input, size_t rows, size_t dim, unsigned bits, const uint16_t* codebook, uint8_t* codes, uint16_t* scales, unsigned threads, char* error, size_t error_size);
int shelf_ngram_decode_rows_f32(const uint8_t* codes, const uint16_t* scales, const uint16_t* codebook, size_t rows, size_t dim, unsigned bits, float* output, char* error, size_t error_size);
int shelf_ngram_verify_bundle(const char* root, char* error, size_t error_size);
#ifdef __cplusplus
}
#endif
