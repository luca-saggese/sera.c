#ifndef Q3_TOKENIZER_H
#define Q3_TOKENIZER_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { uint32_t token_count; uint32_t *tokens; uint64_t prompt_hash, model_hash; } q3_token_batch;
typedef struct q3_vocab_entry q3_vocab_entry;
typedef struct q3_merge_entry q3_merge_entry;
typedef struct {
    char *model_dir;
    q3_vocab_entry *vocab; size_t vocab_cap, vocab_count;
    /* id -> exact GGUF token string. The encode hash table is keyed by string
     * and cannot serve as an id index: distinct ids may share a string, and
     * the table's slot order has nothing to do with token ids. */
    char **id_to_token; size_t id_to_token_count;
    q3_merge_entry *merges; size_t merge_cap, merge_count;
    char **special_text; uint32_t *special_id; size_t special_count;
    uint32_t bos_id, eos_id;
    char *chat_template;
} q3_tokenizer;
typedef struct {
    double wall_ms;
    double file_read_ms;
    double json_scan_ms;
    double vocab_parse_ms;
    double vocab_string_alloc_ms;
    double vocab_hash_build_ms;
    double merges_parse_ms;
    double merges_string_alloc_ms;
    double merges_index_build_ms;
    double special_token_parse_ms;
    double regex_pretokenizer_init_ms;
    double fingerprint_ms;
    double other_ms;
    uint64_t vocab_entries;
    uint64_t merge_entries;
    uint64_t hash_insertions;
    uint64_t hash_lookups;
    uint64_t string_comparisons;
    uint64_t strlen_calls;
    uint64_t string_copy_bytes;
} q3_tokenizer_profile;
#include "q3_gguf.h"
bool q3_tokenizer_init(q3_tokenizer*, const char*, const char*, char*, size_t);
bool q3_tokenizer_init_gguf(q3_tokenizer*, const q3_gguf*, char*, size_t);
void q3_tokenizer_profile_reset(q3_tokenizer_profile *);
void q3_tokenizer_profile_set(q3_tokenizer_profile *);
void q3_tokenizer_profile_note_hash_insertion(void);
void q3_tokenizer_profile_note_hash_lookup(void);
void q3_tokenizer_profile_note_string_comparison(void);
void q3_tokenizer_profile_note_strlen(void);
void q3_tokenizer_profile_note_string_copy(size_t);
bool q3_tokenizer_verify_specials(const q3_tokenizer*, char*, size_t);
void q3_tokenizer_destroy(q3_tokenizer*);
bool q3_tokenizer_encode(const q3_tokenizer*, const char*, bool, q3_token_batch*, char*, size_t);
bool q3_tokenizer_encode_chat_json(const q3_tokenizer*, const char*, bool, bool, q3_token_batch*, char*, size_t);
bool q3_tokenizer_decode(const q3_tokenizer*, const uint32_t*, size_t,
                          char**, size_t*, char*, size_t);
void q3_token_batch_free(q3_token_batch*);
#ifdef __cplusplus
}
#endif
#endif
