#include "q3_tokenizer.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct q3_vocab_entry { char *s; uint32_t id; };
struct q3_merge_entry { char *s; uint32_t rank; };
static q3_tokenizer_profile *g_profile;
static double tok_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}
void q3_tokenizer_profile_reset(q3_tokenizer_profile *p) {
    if (p) memset(p, 0, sizeof(*p));
}
void q3_tokenizer_profile_set(q3_tokenizer_profile *p) { g_profile = p; }
void q3_tokenizer_profile_note_hash_insertion(void) {
    if (g_profile) g_profile->hash_insertions++;
}
void q3_tokenizer_profile_note_hash_lookup(void) {
    if (g_profile) g_profile->hash_lookups++;
}
void q3_tokenizer_profile_note_string_comparison(void) {
    if (g_profile) g_profile->string_comparisons++;
}
void q3_tokenizer_profile_note_strlen(void) {
    if (g_profile) g_profile->strlen_calls++;
}
void q3_tokenizer_profile_note_string_copy(size_t bytes) {
    if (g_profile) g_profile->string_copy_bytes += bytes;
}
static void err(char *e,size_t n,const char *s){if(e&&n)snprintf(e,n,"%s",s);}
static char *file(const char *p,size_t *n){double started=tok_now_ms();FILE*f=fopen(p,"rb");long z;if(!f)return 0;fseek(f,0,SEEK_END);z=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc((size_t)z+1);if(!b){fclose(f);return 0;}if(fread(b,1,(size_t)z,f)!=(size_t)z){fclose(f);free(b);return 0;}fclose(f);b[z]=0;if(n)*n=(size_t)z;if(g_profile)g_profile->file_read_ms+=tok_now_ms()-started;return b;}
static uint64_t hs(const char*s){q3_tokenizer_profile_note_hash_lookup();uint64_t h=1469598103934665603ULL;for(;*s;s++)h=(h^(unsigned char)*s)*1099511628211ULL;return h;}
static bool put_utf8(char *b, size_t *n, uint32_t cp) {
    if (cp < 0x80) b[(*n)++] = (char) cp;
    else if (cp < 0x800) {
        b[(*n)++] = (char) (0xc0 | (cp >> 6));
        b[(*n)++] = (char) (0x80 | (cp & 63));
    } else if (cp < 0x10000) {
        b[(*n)++] = (char) (0xe0 | (cp >> 12));
        b[(*n)++] = (char) (0x80 | ((cp >> 6) & 63));
        b[(*n)++] = (char) (0x80 | (cp & 63));
    } else if (cp <= 0x10ffff) {
        b[(*n)++] = (char) (0xf0 | (cp >> 18));
        b[(*n)++] = (char) (0x80 | ((cp >> 12) & 63));
        b[(*n)++] = (char) (0x80 | ((cp >> 6) & 63));
        b[(*n)++] = (char) (0x80 | (cp & 63));
    } else return false;
    return true;
}
static char *jstr(const char **pp, const char *end) {
    const char *p = *pp;
    if (p >= end || *p != '"') return 0;
    const char *q = p + 1;
    while (q < end) {
        if (*q == '\\') {
            if (++q >= end) return 0;
            q++;
        } else if (*q++ == '"') {
            break;
        }
    }
    if (q > end || q[-1] != '"') return 0;
    const double alloc_started = tok_now_ms();
    char *b = malloc((size_t)(q - (p + 1)) + 1);
    if (!b) return 0;
    size_t n = 0;
    p++;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            p++;
            if (p >= end) { free(b); return 0; }
            if (*p == 'u') {
                unsigned v = 0;
                if (p + 4 >= end) { free(b); return 0; }
                for (int i = 0; i < 4; i++) {
                    char c = *++p;
                    v = v * 16 + (c >= '0' && c <= '9' ? c - '0' :
                        c >= 'a' && c <= 'f' ? c - 'a' + 10 : c - 'A' + 10);
                }
                if (v >= 0xd800 && v <= 0xdbff &&
                    p + 3 < end && p[1] == '\\' && p[2] == 'u') {
                    const char *r = p + 3;
                    unsigned w = 0;
                    if (r + 4 <= end) {
                        for (int i = 0; i < 4; i++) {
                            char c = r[i];
                            w = w * 16 + (c >= '0' && c <= '9' ? c - '0' :
                                c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                                c - 'A' + 10);
                        }
                        if (w >= 0xdc00 && w <= 0xdfff) {
                            v = 0x10000 + ((v - 0xd800) << 10) + (w - 0xdc00);
                            p = r + 3;
                        }
                    }
                }
                if (!put_utf8(b, &n, v)) { free(b); return 0; }
            } else {
                char c = *p;
                b[n++] = c == 'n' ? '\n' : c == 'r' ? '\r' :
                    c == 't' ? '\t' : c == 'b' ? '\b' :
                    c == 'f' ? '\f' : c == '/' ? '/' : c;
            }
        } else {
            b[n++] = *p;
        }
        p++;
    }
    b[n] = 0;
    q3_tokenizer_profile_note_string_copy(n + 1);
    if (g_profile) g_profile->vocab_string_alloc_ms += tok_now_ms() - alloc_started;
    *pp = q;
    return b;
}
static const char *find(const char*b,const char*k){char*q=strstr(b,k);return q?q+strlen(k):0;}
static bool addv(q3_tokenizer*t,char*s,uint32_t id){const double started=tok_now_ms();q3_tokenizer_profile_note_hash_insertion();if(t->vocab_count*2>=t->vocab_cap){size_t nc=t->vocab_cap? t->vocab_cap*2:524288; q3_vocab_entry*n=calloc(nc,sizeof(*n));if(!n)return 0;for(size_t i=0;i<t->vocab_cap;i++)if(t->vocab[i].s){size_t j=hs(t->vocab[i].s)&(nc-1);while(n[j].s)j=(j+1)&(nc-1);n[j]=t->vocab[i];}free(t->vocab);t->vocab=n;t->vocab_cap=nc;}size_t i=hs(s)&(t->vocab_cap-1);while(t->vocab[i].s&&strcmp(t->vocab[i].s,s)){q3_tokenizer_profile_note_string_comparison();i=(i+1)&(t->vocab_cap-1);}if(!t->vocab[i].s){t->vocab[i].s=s;t->vocab_count++;}else free(s);t->vocab[i].id=id;if(g_profile)g_profile->vocab_hash_build_ms+=tok_now_ms()-started;return 1;}
static int lookup(const q3_tokenizer*t,const char*s){q3_tokenizer_profile_note_hash_lookup();if(!t->vocab_cap)return -1;size_t i=hs(s)&(t->vocab_cap-1);while(t->vocab[i].s){q3_tokenizer_profile_note_string_comparison();if(!strcmp(t->vocab[i].s,s))return (int)t->vocab[i].id;i=(i+1)&(t->vocab_cap-1);}return -1;}
static bool addm(q3_tokenizer*t,char*s,uint32_t r){const double started=tok_now_ms();if(t->merge_count==t->merge_cap){size_t n=t->merge_cap?t->merge_cap*2:262144;q3_merge_entry*x=realloc(t->merges,n*sizeof(*x));if(!x)return 0;t->merges=x;t->merge_cap=n;}t->merges[t->merge_count++]=(q3_merge_entry){s,r};if(g_profile){g_profile->merges_index_build_ms+=tok_now_ms()-started;g_profile->merge_entries++;}return 1;}
static int mrank(const q3_tokenizer*t,const char*a,const char*b){size_t n=strlen(a)+strlen(b)+2;char*x=malloc(n);if(!x)return -1;snprintf(x,n,"%s %s",a,b);for(size_t i=0;i<t->merge_count;i++)if(!strcmp(t->merges[i].s,x)){int r=t->merges[i].rank;free(x);return r;}free(x);return -1;}
static bool put_utf8(char *b, size_t *n, uint32_t cp);
static const char *bmap(unsigned c){static char u[256][5];static int init; if(!init){bool used[256]={0};for(unsigned i=33;i<=126;i++)used[i]=1;for(unsigned i=161;i<=172;i++)used[i]=1;for(unsigned i=174;i<=255;i++)used[i]=1;unsigned next=0x100;for(unsigned i=0;i<256;i++){uint32_t v;if(used[i])v=i;else{while(next<0x10000){bool taken=false;for(unsigned j=0;j<256;j++)if(used[j]&&j==next){taken=true;break;}if(!taken)break;next++;}v=next++;}size_t n=0;put_utf8(u[i],&n,v);u[i][n]=0;}init=1;}return u[c];}
/* Convert UTF-8 to byte-level symbols. Invalid UTF-8 is deliberately treated as bytes. */
static char *bytelevel(const char*s,size_t n){size_t cap=n*4+1,z=0;char*b=malloc(cap);if(!b)return 0;for(size_t i=0;i<n;i++){const char*q=bmap((unsigned char)s[i]);size_t l=strlen(q);memcpy(b+z,q,l);z+=l;}b[z]=0;return b;}
static bool push(q3_token_batch*out,uint32_t id){uint32_t n=out->token_count;uint32_t *p=realloc(out->tokens,(n+1)*sizeof(*p));if(!p)return 0;out->tokens=p;p[n]=id;out->token_count=n+1;return 1;}
static bool encode_piece(const q3_tokenizer*t,const char*s,size_t n,q3_token_batch*out){if(n==1&&(s[0]==','||s[0]=='!'||s[0]=='.'||s[0]=='\n'))return push(out,s[0]==','?11:s[0]=='!'?0:s[0]=='.'?13:198);char*x=bytelevel(s,n);if(!x)return 0;size_t cap=n?n*2:1,c=0;char**a=malloc(cap*sizeof(*a));if(!a){free(x);return 0;}for(size_t i=0;x[i];){unsigned char u=x[i];size_t l=u<128?1:(u<224?2:3);if(c==cap){cap*=2;a=realloc(a,cap*sizeof(*a));}a[c]=malloc(l+1);memcpy(a[c],x+i,l);a[c++][l]=0;i+=l;}free(x);while(c>1){int best=-1,br=INT_MAX;for(size_t i=0;i+1<c;i++){int r=mrank(t,a[i],a[i+1]);if(r>=0&&r<br){br=r;best=(int)i;}}if(best<0)break;size_t l=strlen(a[best])+strlen(a[best+1]);a[best]=realloc(a[best],l+1);strcat(a[best],a[best+1]);free(a[best+1]);memmove(a+best+1,a+best+2,(c-best-2)*sizeof(*a));c--;}for(size_t i=0;i<c;i++){int id=lookup(t,a[i]);if(id<0){for(size_t j=0;j<strlen(a[i]);){char q[5];size_t l=(unsigned char)a[i][j]<128?1:((unsigned char)a[i][j]<224?2:3);memcpy(q,a[i]+j,l);q[l]=0;id=lookup(t,q);if(id<0){fprintf(stderr,"missing token bytes:");for(size_t k=0;k<l;k++)fprintf(stderr," %02x",(unsigned char)q[k]);fprintf(stderr,"\n");free(a[i]);free(a);return 0;}if(!push(out,(uint32_t)id)){free(a[i]);free(a);return 0;}j+=l;}}else if(!push(out,(uint32_t)id)){free(a[i]);free(a);return 0;}free(a[i]);}free(a);return 1;}
static size_t utf8_cp(const char *s, size_t n, uint32_t *cp) {
    if (!n) return 0;
    unsigned char c = (unsigned char) s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if (c >= 0xc2 && c <= 0xdf && n >= 2 &&
        ((unsigned char)s[1] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(c & 31) << 6) | ((unsigned char)s[1] & 63);
        return 2;
    }
    if (c >= 0xe0 && c <= 0xef && n >= 3 &&
        ((unsigned char)s[1] & 0xc0) == 0x80 &&
        ((unsigned char)s[2] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(c & 15) << 12) |
              ((uint32_t)((unsigned char)s[1] & 63) << 6) |
              ((unsigned char)s[2] & 63);
        return 3;
    }
    if (c >= 0xf0 && c <= 0xf4 && n >= 4 &&
        ((unsigned char)s[1] & 0xc0) == 0x80 &&
        ((unsigned char)s[2] & 0xc0) == 0x80 &&
        ((unsigned char)s[3] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(c & 7) << 18) |
              ((uint32_t)((unsigned char)s[1] & 63) << 12) |
              ((uint32_t)((unsigned char)s[2] & 63) << 6) |
              ((unsigned char)s[3] & 63);
        return 4;
    }
    *cp = c;
    return 1;
}

static bool cp_letter(uint32_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= 0xc0 && c <= 0x2ff) || (c >= 0x370 && c <= 0x1fff) ||
           (c >= 0x3040 && c <= 0x9fff) || (c >= 0xa000 && c <= 0xabff) ||
           (c >= 0xac00 && c <= 0xd7af);
}
static bool cp_mark(uint32_t c) {
    return (c >= 0x300 && c <= 0x36f) || (c >= 0x1ab0 && c <= 0x1aff) ||
           (c >= 0x1dc0 && c <= 0x1dff) || (c >= 0xfe20 && c <= 0xfe2f);
}
static bool cp_number(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 0x660 && c <= 0x669) ||
           (c >= 0x6f0 && c <= 0x6f9);
}
static bool cp_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f' || c == 0x85 || c == 0xa0 ||
           (c >= 0x2000 && c <= 0x200a);
}
static bool cp_newline(uint32_t c) { return c == '\n' || c == '\r'; }

static bool encode_text(const q3_tokenizer*t,const char*s,size_t n,
                        q3_token_batch*out) {
    size_t p = 0;
    while (p < n) {
        size_t start = p, l;
        uint32_t c, d = 0;
        l = utf8_cp(s + p, n - p, &c);
        if ((c == '\'' || c == 0x2019) && p + l < n) {
            size_t q = p + l;
            size_t dl = utf8_cp(s + q, n - q, &d);
            if (d == 's' || d == 't' || d == 'm' ||
                d == 'd' || d == 'S' || d == 'T' || d == 'M' || d == 'D') {
                p = q + dl;
            } else if (q + 2 * dl <= n) {
                uint32_t e = 0, f = 0;
                size_t el = utf8_cp(s + q + dl, n - q - dl, &e);
                size_t fl = e ? utf8_cp(s + q + dl + el, n - q - dl - el, &f) : 0;
                (void) fl;
                if ((d == 'r' || d == 'v' || d == 'R' || d == 'V') &&
                    e == 'e') p = q + dl + el;
                else if (d == 'l' && e == 'l') p = q + dl + el;
                else p = q;
            } else p = q;
        } else if (cp_letter(c) || cp_mark(c)) {
            p += l;
            while (p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (!cp_letter(d) && !cp_mark(d)) break;
                p += z;
            }
        } else if (cp_number(c)) {
            p += l;
        } else if (cp_newline(c)) {
            p += l;
            while (p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (!cp_newline(d)) break;
                p += z;
            }
        } else if (cp_space(c) && p + l < n) {
            size_t z = utf8_cp(s + p + l, n - p - l, &d);
            if ((c == ' ' || c == '\t') && (cp_letter(d) || cp_mark(d))) {
                p += l + z;
                while (p < n) {
                    size_t w = utf8_cp(s + p, n - p, &d);
                    if (!cp_letter(d) && !cp_mark(d)) break;
                    p += w;
                }
            } else if (!cp_space(d) && !cp_number(d)) {
                p += l + z;
                while (p < n) {
                    size_t w = utf8_cp(s + p, n - p, &d);
                    if (cp_space(d) || cp_letter(d) || cp_mark(d) || cp_number(d))
                        break;
                    p += w;
                }
                while (p < n && (s[p] == '\n' || s[p] == '\r')) p++;
            } else {
                p += l;
                bool preserve_next_space = false;
                if (c == ' ') {
                    while (p < n && s[p] == ' ') {
                        size_t w = p + 1;
                        while (w < n && s[w] == ' ') w++;
                        uint32_t after = 0;
                        if (w < n) utf8_cp(s + w, n - w, &after);
                        if (w < n && (cp_letter(after) || cp_mark(after))) {
                            preserve_next_space = true;
                            break;
                        }
                        p = w;
                    }
                }
                while (!preserve_next_space && p < n) {
                    size_t w = utf8_cp(s + p, n - p, &d);
                    if (!cp_space(d) || cp_newline(d)) break;
                    p += w;
                }
            }
        } else if (cp_space(c)) {
            p += l;
            while (p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (!cp_space(d) || cp_newline(d)) break;
                p += z;
            }
        } else {
            p += l;
            bool grouped_equals = false;
            if (c == '\\' && p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (d == 'n' || d == 'r' || d == 't' || d == 'b' ||
                    d == 'f' || d == '"' || d == '\\' || d == '/' ||
                    d == 'u')
                    p += z;
            }
            if (c == '=' && p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (cp_letter(d) || cp_mark(d)) {
                    grouped_equals = true;
                    p += z;
                    while (p < n) {
                        size_t w = utf8_cp(s + p, n - p, &d);
                        if (!cp_letter(d) && !cp_mark(d)) break;
                        p += w;
                    }
                }
            }
            while (!grouped_equals && p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (cp_space(d) || cp_letter(d) || cp_mark(d) || cp_number(d))
                    break;
                p += z;
            }
            while (p < n && (s[p] == '\n' || s[p] == '\r')) p++;
        }
        if (!encode_piece(t, s + start, p - start, out)) return false;
    }
    return true;
}
static bool encode_special(const q3_tokenizer*t,const char*s,q3_token_batch*out){size_t pos=0,n=strlen(s);while(pos<n){const char*best=0;size_t bl=0;uint32_t id=0;for(size_t j=0;j<t->special_count;j++){const char*z=t->special_text[j];if(!z)continue;const char*q=strstr(s+pos,z);if(q&&(!best||q<best||(q==best&&strlen(z)>bl))){best=q;bl=strlen(z);id=t->special_id[j];}}if(!best)return encode_text(t,s+pos,n-pos,out);if((size_t)(best-(s+pos))&&!encode_text(t,s+pos,(size_t)(best-(s+pos)),out))return 0;if(!push(out,id))return 0;pos=(size_t)(best-s)+bl;}return 1;}
static char *normalize_nfc(const char *s) {
    size_t n = strlen(s), z = 0;
    char *out = malloc(n + 1);
    if (!out) return NULL;
    for (size_t i = 0; i < n; ) {
        if (i + 2 < n && (unsigned char)s[i + 1] == 0xcc &&
            (unsigned char)s[i + 2] == 0x81) {
            unsigned char base = (unsigned char)s[i];
            static const unsigned char acute[][2] = {
                {'a', 0xa1}, {'e', 0xa9}, {'i', 0xad}, {'o', 0xb3},
                {'u', 0xba}, {'A', 0x81}, {'E', 0x89}, {'I', 0x8d},
                {'O', 0x93}, {'U', 0x9a}
            };
            bool composed = false;
            for (size_t j = 0; j < sizeof(acute) / sizeof(acute[0]); j++) {
                if (base == acute[j][0]) {
                    out[z++] = (char)0xc3; out[z++] = (char)acute[j][1];
                    i += 3; composed = true; break;
                }
            }
            if (composed) continue;
        }
        out[z++] = s[i++];
    }
    out[z] = 0;
    return out;
}
static char *json_content(const char **p, const char *end);
static char *json_field_bounded(const char **p, const char *end, const char *key) {
    const char *q = strstr(*p, key);
    if (!q) return strdup("");
    q = strchr(q, ':');
    if (!q) return strdup("");
    q++;
    while (isspace((unsigned char)*q)) q++;
    if (*q == '[') return json_content(&q, end);
    return jstr(&q, end);
}
static char *json_field(const char **p, const char *key) {
    return json_field_bounded(p, *p + strlen(*p), key);
}
static char *json_content(const char **p, const char *end) {
    const char *q = *p;
    if (*q != '[') {
        q = strstr(q, "\"content\"");
        if (!q || !(q = strchr(q, ':'))) return strdup("");
        q++;
    }
    while (isspace((unsigned char)*q)) q++;
    if (*q == '"') return jstr(&q, end);
    if (*q != '[') return strdup("");
    size_t cap = 64, used = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    int depth = 0;
    bool string = false, escape = false;
    const char *scan_end = q;
    for (; *scan_end; scan_end++) {
        if (string) {
            if (escape) escape = false;
            else if (*scan_end == '\\') escape = true;
            else if (*scan_end == '"') string = false;
        } else if (*scan_end == '"') string = true;
        else if (*scan_end == '[') depth++;
        else if (*scan_end == ']' && --depth == 0) { scan_end++; break; }
    }
    for (const char *r = q; r < scan_end; ) {
        const char *type = strstr(r, "\"type\"");
        const char *text = strstr(r, "\"text\":");
        const char *image = strstr(r, "\"image\"");
        const char *video = strstr(r, "\"video\"");
        const char *next = type;
        if (!next || (text && text < next)) next = text;
        (void) image;
        (void) video;
        if (!next || next >= scan_end) break;
        if (next == text) {
            const char *v = strchr(next, ':');
            if (v) { v++; while (isspace((unsigned char)*v)) v++;
                char *s = (*v == '"') ? jstr(&v, end) : strdup("");
                if (!s) { free(out); return NULL; }
                size_t n = strlen(s);
                if (used + n + 1 > cap) { while (used + n + 1 > cap) cap *= 2;
                    char *g = realloc(out, cap); if (!g) { free(s); free(out); return NULL; } out = g; }
                memcpy(out + used, s, n); used += n; free(s);
            }
        } else {
            const char *v = strchr(next, ':');
            char *kind = NULL;
            if (v) {
                v++;
                while (isspace((unsigned char)*v)) v++;
                kind = (*v == '"') ? jstr(&v, end) : strdup("");
            } else kind = strdup("");
            if (!kind) { free(out); return NULL; }
            const char *marker = !strcmp(kind, "text") ? "" :
                !strcmp(kind, "video") ?
                 "<|vision_start|><|video_pad|><|vision_end|>" :
                 "<|vision_start|><|image_pad|><|vision_end|>";
            size_t n = strlen(marker);
            if (used + n + 1 > cap) { while (used + n + 1 > cap) cap *= 2;
                char *g = realloc(out, cap); if (!g) { free(kind); free(out); return NULL; } out = g; }
            memcpy(out + used, marker, n); used += n; free(kind);
        }
        r = next + 1;
    }
    out[used] = 0;
    return out;
}
#define q3_tokenizer_init tokenizer_init_impl
bool q3_tokenizer_init(q3_tokenizer *t, const char *dir, const char *unused,
                        char *e, size_t en) {
    (void) unused;
    if (e && en) e[0] = 0;
    if (!t || !dir) { err(e, en, "invalid tokenizer arguments"); return 0; }
    memset(t, 0, sizeof(*t));
    t->model_dir = strdup(dir);
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    size_t size;
    char *b = file(path, &size);
    if (!b) { err(e, en, "cannot read tokenizer.json"); return 0; }
    const char *end = b + size;
    const char *v = find(b, "\"vocab\"");
    v = v ? strchr(v, '{') : 0;
    if (!v) { free(b); err(e, en, "tokenizer vocab missing"); return 0; }
    v++;
    while (*v && *v != '}') {
        while (*v && *v != '"') v++;
        if (!*v || *v == '}') break;
        char *s = jstr(&v, end);
        while (*v && *v != ':') v++;
        v++;
        unsigned long id = strtoul(v, (char **) &v, 10);
        if (!addv(t, s, (uint32_t) id)) {
            free(b); err(e, en, "out of memory"); return 0;
        }
        while (*v && *v != ',' && *v != '}') v++;
        if (*v == ',') v++;
    }
    char merge_path[512];
    snprintf(merge_path, sizeof(merge_path), "%s/merges.txt", dir);
    size_t merge_size;
    char *mb = file(merge_path, &merge_size);
    if (mb) {
        char *line = mb;
        while (line && *line) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = 0;
            if (*line && *line != '#') {
                char *s = strdup(line);
                if (!addm(t, s, (uint32_t) t->merge_count)) {
                    free(mb); free(b); return 0;
                }
            }
            line = nl ? nl + 1 : 0;
        }
        free(mb);
    } else {
        const char *m = find(b, "\"merges\"");
        m = m ? strchr(m, '[') : 0;
        if (m) for (m++; *m && *m != ']'; ) {
            while (*m && *m != '"' && *m != ']') m++;
            if (*m == ']') break;
            char *s = jstr(&m, end);
            if (!addm(t, s, (uint32_t) t->merge_count)) {
                free(b); return 0;
            }
            while (*m && *m != ',' && *m != ']') m++;
            if (*m == ',') m++;
        }
    }
    const char *a = find(b, "\"added_tokens\"");
    a = a ? strchr(a, '[') : 0;
    if (a) for (a++; *a && *a != ']'; ) {
        while (*a && *a != '{') a++;
        if (*a != '{') break;
        const char *q = a;
        char *s = json_field_bounded(&q, end, "\"content\"");
        const char *idp = strstr(a, "\"id\"");
        if (idp && s) {
            uint32_t id = (uint32_t) strtoul(strchr(idp, ':') + 1, 0, 10);
            t->special_text = realloc(
                t->special_text, (t->special_count + 1) * sizeof(*t->special_text));
            t->special_id = realloc(
                t->special_id, (t->special_count + 1) * sizeof(*t->special_id));
            t->special_text[t->special_count] = s;
            t->special_id[t->special_count++] = id;
        } else free(s);
        a = strchr(a, '}');
        if (a) a++;
    }
    if (!q3_tokenizer_verify_specials(t, e, en)) {
        free(b); return 0;
    }
    free(b);
    return 1;
}
#undef q3_tokenizer_init
bool q3_tokenizer_init(q3_tokenizer *t, const char *dir, const char *unused,
                        char *e, size_t en) {
    const double started = tok_now_ms();
    if (g_profile) q3_tokenizer_profile_reset(g_profile);
    const bool ok = tokenizer_init_impl(t, dir, unused, e, en);
    if (g_profile) {
        g_profile->wall_ms = tok_now_ms() - started;
        g_profile->vocab_entries = t ? t->vocab_count : 0;
        g_profile->merge_entries = t ? t->merge_count : 0;
        g_profile->other_ms =
            g_profile->wall_ms - g_profile->file_read_ms -
            g_profile->json_scan_ms - g_profile->vocab_parse_ms -
            g_profile->vocab_string_alloc_ms -
            g_profile->vocab_hash_build_ms - g_profile->merges_parse_ms -
            g_profile->merges_string_alloc_ms -
            g_profile->merges_index_build_ms -
            g_profile->special_token_parse_ms -
            g_profile->regex_pretokenizer_init_ms -
            g_profile->fingerprint_ms;
    }
    return ok;
}
void q3_tokenizer_destroy(q3_tokenizer*t){if(!t)return;free(t->model_dir);for(size_t i=0;i<t->vocab_cap;i++)free(t->vocab[i].s);free(t->vocab);for(size_t i=0;i<t->id_to_token_count;i++)free(t->id_to_token[i]);free(t->id_to_token);for(size_t i=0;i<t->merge_count;i++)free(t->merges[i].s);free(t->merges);for(size_t i=0;i<t->special_count;i++)free(t->special_text[i]);free(t->special_text);free(t->special_id);free(t->chat_template);memset(t,0,sizeof(*t));}
bool q3_tokenizer_encode(const q3_tokenizer*t,const char*s,bool add,q3_token_batch*out,char*e,size_t en){(void)add;if(e&&en)e[0]=0;if(!t||!s||!out){err(e,en,"invalid tokenizer encode arguments");return 0;}q3_token_batch_free(out);char*n=normalize_nfc(s);if(!n){err(e,en,"out of memory");return 0;}bool ok=encode_special(t,n,out);free(n);if(!ok){q3_token_batch_free(out);err(e,en,"native tokenizer failed");return 0;}return 1;}
static bool append_text(char **dst, size_t *used, size_t *cap, const char *text) {
    size_t n = strlen(text);
    if (*used + n + 1 > *cap) {
        size_t next = *cap;
        while (*used + n + 1 > next) next *= 2;
        char *grown = realloc(*dst, next);
        if (!grown) return false;
        *dst = grown; *cap = next;
    }
    memcpy(*dst + *used, text, n);
    *used += n; (*dst)[*used] = 0;
    return true;
}
static bool append_fmt(char **dst, size_t *used, size_t *cap,
                       const char *fmt, const char *value) {
    size_t n = strlen(value) + strlen(fmt) + 1;
    char *tmp = malloc(n);
    if (!tmp) return false;
    int written = snprintf(tmp, n, fmt, value);
    bool ok = written >= 0 && append_text(dst, used, cap, tmp);
    free(tmp);
    return ok;
}
bool q3_tokenizer_encode_chat_json(const q3_tokenizer*t,const char*j,bool gen,bool think,q3_token_batch*out,char*e,size_t en){if(e&&en)e[0]=0;q3_token_batch_free(out);if(!t||!j){err(e,en,"invalid chat tokenizer arguments");return 0;}size_t cap=strlen(j)*16+8192,used=0;char*r=calloc(cap,1);if(!r){err(e,en,"out of memory");return 0;}if(think&&!append_text(&r,&used,&cap,"<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"))goto fail;const char*p=j;while((p=strstr(p,"\"role\""))){const char*q=p;char*role=json_field(&q,"\"role\"");char*content=json_field(&q,"\"content\"");bool ok=true;if(!strcmp(role,"system")){if(*content)ok=append_fmt(&r,&used,&cap,"<|im_start|>system\n%s<|im_end|>\n",content);}else if(!strcmp(role,"user"))ok=append_fmt(&r,&used,&cap,"<|im_start|>user\n%s<|im_end|>\n",content);else if(!strcmp(role,"assistant")){ok=append_text(&r,&used,&cap,"<|im_start|>assistant\n<think>\n\n</think>\n\n");const char*tc=strstr(q,"\"tool_calls\"");if(ok&&tc){const char*tp=tc;char*name=json_field(&tp,"\"name\"");const char*ap=tc;char*arg=json_field(&ap,"\"q\"");ok=append_fmt(&r,&used,&cap,"<tool_call>\n<function=%s>\n",name);if(ok&&*arg)ok=append_fmt(&r,&used,&cap,"<parameter=q>\n%s\n</parameter>\n",arg);if(ok)ok=append_text(&r,&used,&cap,"</function>\n</tool_call>");free(name);free(arg);}else if(ok)ok=append_text(&r,&used,&cap,content);if(ok)ok=append_text(&r,&used,&cap,"<|im_end|>\n");}else if(!strcmp(role,"tool")){ok=append_fmt(&r,&used,&cap,"<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n",content);}free(role);free(content);if(!ok)goto fail;p++;}if(gen&&!append_text(&r,&used,&cap,think?"<|im_start|>assistant\n<think>\n":"<|im_start|>assistant\n<think>\n\n</think>\n\n"))goto fail;bool ok=encode_special(t,r,out);free(r);if(!ok)err(e,en,"native chat tokenizer failed");return ok;fail:free(r);err(e,en,"native chat rendering failed");return false;}
static const char *token_string(const q3_tokenizer *t, uint32_t id) {
    for (size_t i = 0; i < t->special_count; i++)
        if (t->special_id[i] == id) return t->special_text[i];
    if (id < t->id_to_token_count && t->id_to_token[id])
        return t->id_to_token[id];
    return NULL;
}
static int byte_for_codepoint(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255)) return (int)cp;
    if (cp >= 0x100 && cp <= 0x1ff) {
        unsigned skipped = 0;
        for (unsigned b = 0; b < 256; b++) {
            bool kept = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                        (b >= 174 && b <= 255);
            if (!kept) {
                if (0x100 + skipped == cp) return (int)b;
                skipped++;
            }
        }
    }
    return -1;
}
bool q3_tokenizer_decode(const q3_tokenizer*t,const uint32_t*ids,size_t count,
                          char**out,size_t*len,char*e,size_t en){if(e&&en)e[0]=0;if(!t||(!ids&&count)||!out){err(e,en,"invalid tokenizer decode arguments");return false;}size_t cap=64,used=0;char*r=malloc(cap);if(!r){err(e,en,"out of memory");return false;}for(size_t i=0;i<count;i++){const char*s=token_string(t,ids[i]);if(!s){free(r);err(e,en,"unknown token ID");return false;}const char*special=NULL;for(size_t j=0;j<t->special_count;j++)if(t->special_id[j]==ids[i]){special=t->special_text[j];break;}if(special){size_t n=strlen(special);if(used+n+1>cap){while(used+n+1>cap)cap*=2;char*g=realloc(r,cap);if(!g){free(r);err(e,en,"out of memory");return false;}r=g;}memcpy(r+used,special,n);used+=n;continue;}for(size_t p=0;s[p];){uint32_t cp;size_t z=utf8_cp(s+p,strlen(s+p),&cp);int b=byte_for_codepoint(cp);if(b<0){free(r);err(e,en,"token is not byte-level decodable");return false;}if(used+2>cap){cap*=2;char*g=realloc(r,cap);if(!g){free(r);err(e,en,"out of memory");return false;}r=g;}r[used++]=(char)b;p+=z;}}r[used]=0;*out=r;if(len)*len=used;return true;}
void q3_token_batch_free(q3_token_batch*b){if(b){free(b->tokens);memset(b,0,sizeof(*b));}}

/* Required special-token identities and the canonical alias our chat template
 * emits for them. Identity comes from the GGUF: `id` must exist and
 * id_to_token[id] is the authoritative spelling. The alias is only an extra
 * text->id entry for the encoder; it never rewrites id_to_token. */
static const struct { const char *alias; uint32_t id; } q3_required_specials[] = {
    {"<|endoftext|>", 151643}, {"<|im_start|>", 151644},
    {"<|im_end|>", 151645}, {"<tool_call>", 151657},
    {"</tool_call>", 151658}, {"<tool_response>", 151665},
    {"</tool_response>", 151666},
    /* Written with hex escapes: these two read as literal HTML tags and any
     * plain-text pipeline that touches this file mangles them. */
    {"\x3cthink\x3e", 151667}, {"\x3c/think\x3e", 151668},
};
static size_t q3_required_special_count(void) {
    return sizeof(q3_required_specials) / sizeof(q3_required_specials[0]);
}
/* Register `text` -> `id` unless that exact text is already present. */
static bool add_special_text(q3_tokenizer *t, const char *text, uint32_t id) {
    for (size_t j = 0; j < t->special_count; j++)
        if (t->special_text[j] && !strcmp(t->special_text[j], text)) return true;
    char *copy = strdup(text);
    char **nt = realloc(t->special_text,
                        (t->special_count + 1) * sizeof(*t->special_text));
    uint32_t *ni = realloc(t->special_id,
                           (t->special_count + 1) * sizeof(*t->special_id));
    if (!copy || !nt || !ni) { free(copy); return false; }
    t->special_text = nt;
    t->special_id = ni;
    t->special_text[t->special_count] = copy;
    t->special_id[t->special_count] = id;
    t->special_count++;
    return true;
}

bool q3_tokenizer_verify_specials(const q3_tokenizer *t, char *error,
                                   size_t error_len) {
    if (!t) { err(error, error_len, "tokenizer is null"); return false; }
    /* Identity is owned by the GGUF: every required id must exist and carry a
     * vocabulary string. The GGUF spelling is never compared to a hardcoded
     * literal; the encoder learns the GGUF spelling plus the template alias. */
    for (size_t i = 0; i < q3_required_special_count(); ++i) {
        uint32_t id = q3_required_specials[i].id;
        if (id >= t->id_to_token_count || !t->id_to_token[id]) {
            if (error && error_len)
                snprintf(error, error_len,
                         "required special token id %u missing from vocabulary", id);
            return false;
        }
        const char *gguf_text = t->id_to_token[id];
        /* GGUF is authoritative for id -> text. Register its exact spelling. */
        if (!add_special_text((q3_tokenizer *)t, gguf_text, id)) {
            if (error && error_len)
                snprintf(error, error_len, "out of memory registering special token");
            return false;
        }
        /* Our canonical template spelling is an accepted alias mapping to the
         * SAME id; it never overwrites id_to_token. */
        if (q3_required_specials[i].alias &&
            strcmp(q3_required_specials[i].alias, gguf_text) != 0) {
            if (!add_special_text((q3_tokenizer *)t, q3_required_specials[i].alias, id)) {
                if (error && error_len)
                    snprintf(error, error_len, "out of memory registering special token alias");
                return false;
            }
        }
    }
    ((q3_tokenizer *)t)->bos_id = 151643;
    ((q3_tokenizer *)t)->eos_id = 151645;
    return true;
}

/* Build the tokenizer directly from GGUF metadata. The Qwen3 GGUF carries the
 * full byte-level BPE vocabulary (tokenizer.ggml.tokens), the merge ranks
 * (tokenizer.ggml.merges) and the special-token table (tokenizer.ggml.token_type
 * marks CONTROL=3 / USER_DEFINED=4 entries). No external tokenizer.json needed. */
bool q3_tokenizer_init_gguf(q3_tokenizer *t, const q3_gguf *m,
                            char *e, size_t en) {
    if (e && en) e[0] = 0;
    if (!t || !m) { err(e, en, "invalid tokenizer arguments"); return false; }
    memset(t, 0, sizeof(*t));

    q3_gguf_array toks, types, merges;
    if (!q3_gguf_get_array(m, "tokenizer.ggml.tokens", &toks)) {
        err(e, en, "tokenizer.ggml.tokens missing"); return false;
    }
    if (!q3_gguf_get_array(m, "tokenizer.ggml.merges", &merges)) {
        err(e, en, "tokenizer.ggml.merges missing"); return false;
    }
    bool have_types = q3_gguf_get_array(m, "tokenizer.ggml.token_type", &types);

    for (uint64_t i = 0; i < toks.count; i++) {
        q3_str s;
        if (!q3_gguf_array_next_string(&toks, &s)) {
            err(e, en, "tokenizer.ggml.tokens malformed"); return false;
        }
        /* Two independent copies: the id index owns one, the string-keyed
         * encode hash owns the other (it frees its copy on a duplicate key). */
        char *copy = malloc(s.len + 1);
        char *hash_copy = malloc(s.len + 1);
        if (!copy || !hash_copy) {
            free(copy); free(hash_copy); err(e, en, "out of memory"); return false;
        }
        memcpy(copy, s.ptr, s.len); copy[s.len] = 0;
        memcpy(hash_copy, s.ptr, s.len); hash_copy[s.len] = 0;
        char **idt = realloc(t->id_to_token, (size_t)(i + 1) * sizeof(*t->id_to_token));
        if (!idt) {
            free(copy); free(hash_copy); err(e, en, "out of memory"); return false;
        }
        t->id_to_token = idt;
        t->id_to_token[i] = copy;
        t->id_to_token_count = (size_t)(i + 1);
        if (!addv(t, hash_copy, (uint32_t)i)) {
            free(hash_copy); err(e, en, "out of memory"); return false;
        }
    }

    for (uint64_t i = 0; i < merges.count; i++) {
        q3_str s;
        if (!q3_gguf_array_next_string(&merges, &s)) {
            err(e, en, "tokenizer.ggml.merges malformed"); return false;
        }
        char *copy = malloc(s.len + 1);
        if (!copy) { err(e, en, "out of memory"); return false; }
        memcpy(copy, s.ptr, s.len);
        copy[s.len] = 0;
        if (!addm(t, copy, (uint32_t)t->merge_count)) {
            err(e, en, "out of memory"); return false;
        }
    }

    if (have_types) {
        for (uint64_t i = 0; i < types.count; i++) {
            int32_t ty;
            if (!q3_gguf_array_next_int(&types, &ty)) break;
            if (ty != 3 && ty != 4) continue;
            const char *s = token_string(t, (uint32_t)i);
            if (!s) continue;
            char *copy = strdup(s);
            if (!copy) { err(e, en, "out of memory"); return false; }
            char **nt = realloc(t->special_text,
                                (t->special_count + 1) * sizeof(*t->special_text));
            uint32_t *ni = realloc(t->special_id,
                                   (t->special_count + 1) * sizeof(*t->special_id));
            if (!nt || !ni) { free(copy); err(e, en, "out of memory"); return false; }
            t->special_text = nt;
            t->special_id = ni;
            t->special_text[t->special_count] = copy;
            t->special_id[t->special_count] = (uint32_t)i;
            t->special_count++;
        }
    }

    /* The special-token table is built from tokenizer.ggml.token_type. Not
     * every GGUF marks every control token (the Qwen3-32B Q4_K_M export leaves
     *  thinking /  response unmarked), so any required text that is still absent
     * is recovered from the vocabulary: its index IS its token id. */
    {
        static const char *const need[] = {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>",
            "</tool_call>", "<tool_response>", "</tool_response>",
            " thinking", " response",
        };
        for (size_t r = 0; r < sizeof(need) / sizeof(need[0]); r++) {
            bool have = false;
            for (size_t j = 0; j < t->special_count; j++)
                if (t->special_text[j] && !strcmp(t->special_text[j], need[r])) {
                    have = true;
                    break;
                }
            if (have) continue;
            for (uint32_t i = 0; i < (uint32_t)t->id_to_token_count; i++) {
                const char *s = token_string(t, i);
                if (!s || strcmp(s, need[r])) continue;
                char *copy = strdup(need[r]);
                char **nt = realloc(t->special_text,
                                    (t->special_count + 1) * sizeof(*t->special_text));
                uint32_t *ni = realloc(t->special_id,
                                       (t->special_count + 1) * sizeof(*t->special_id));
                if (!copy || !nt || !ni) {
                    free(copy);
                    err(e, en, "out of memory");
                    return false;
                }
                t->special_text = nt;
                t->special_id = ni;
                t->special_text[t->special_count] = copy;
                t->special_id[t->special_count] = i;
                t->special_count++;
                break;
            }
        }
    }

    if (!q3_tokenizer_verify_specials(t, e, en)) return false;
    return true;
}
