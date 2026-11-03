// Dump llama.cpp logits for a token sequence, decoded in chunks (chunk = 1 -> token-by-token decode through the KV
// cache and recurrent state). Usage: phase_logits model.gguf tokens.txt out.f32 [chunk] [fa:0|1] [threads] [kv:f16|f32]
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf tokens.txt out.f32 [chunk] [fa] [threads]\n", argv[0]); return 1; }
    const int chunk   = argc > 4 ? atoi(argv[4]) : 1 << 20;
    const int fa      = argc > 5 ? atoi(argv[5]) : 0;
    const int threads = argc > 6 ? atoi(argv[6]) : 4;
    const bool kv_f32 = argc > 7 && std::string(argv[7]) == "f32";

    std::vector<llama_token> toks;
    { std::ifstream f(argv[2]); long t; while (f >> t) toks.push_back((llama_token) t); }
    const int n = (int) toks.size();

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (kv_f32) { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<float> out((size_t) n * n_vocab);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int s = 0; s < n; s += chunk) {
        const int e = s + chunk < n ? s + chunk : n;
        batch.n_tokens = 0;
        for (int i = s; i < e; ++i) {
            const int j = batch.n_tokens++;
            batch.token[j] = toks[i]; batch.pos[j] = i; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = 1;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", s); return 1; }
        for (int i = s; i < e; ++i) {
            const float * l = llama_get_logits_ith(ctx, i - s);
            std::copy(l, l + n_vocab, out.begin() + (size_t) i * n_vocab);
        }
    }
    FILE * fo = fopen(argv[3], "wb");
    fwrite(out.data(), sizeof(float), out.size(), fo);
    fclose(fo);
    fprintf(stderr, "wrote %d x %d logits (chunk %d, fa %d, kv %s)\n", n, n_vocab, chunk, fa, kv_f32 ? "f32" : "f16");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
