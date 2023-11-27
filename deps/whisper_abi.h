#ifndef WHISPER_ABI_H
#define WHISPER_ABI_H

#include "whisper.h"

#ifdef WHISPER_SHARED
#    ifdef _WIN32
#        ifdef WHISPER_BUILD
#            define WHISPER_API __declspec(dllexport)
#        else
#            define WHISPER_API __declspec(dllimport)
#        endif
#    else
#        define WHISPER_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define WHISPER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif


//==============================whisper_context_params===================================================
WHISPER_API void whisper_context_params_set_use_gpu(struct whisper_context_params *params, bool use_gpu);

//==============================whisper_full_params=======================================================

WHISPER_API void whisper_full_params_set_strategy(struct whisper_full_params *params, const char *strategy);

WHISPER_API void whisper_full_params_set_n_threads(struct whisper_full_params *params, int n_threads);
WHISPER_API void whisper_full_params_set_n_max_text_ctx(struct whisper_full_params *params, int n_max_text_ctx);
WHISPER_API void whisper_full_params_set_offset_ms(struct whisper_full_params *params, int offset_ms);
WHISPER_API void whisper_full_params_set_duration_ms(struct whisper_full_params *params, int duration_ms);

WHISPER_API void whisper_full_params_set_translate(struct whisper_full_params *params, bool translate);
WHISPER_API void whisper_full_params_set_no_context(struct whisper_full_params *params, bool no_context);
WHISPER_API void whisper_full_params_set_no_timestamps(struct whisper_full_params *params, bool no_timestamps);
WHISPER_API void whisper_full_params_set_single_segment(struct whisper_full_params *params, bool single_segment);
WHISPER_API void whisper_full_params_set_print_special(struct whisper_full_params *params, bool print_special);
WHISPER_API void whisper_full_params_set_print_progress(struct whisper_full_params *params, bool print_progress);
WHISPER_API void whisper_full_params_set_print_realtime(struct whisper_full_params *params, bool print_realtime);
WHISPER_API void whisper_full_params_set_print_timestamps(struct whisper_full_params *params, bool print_timestamps);

WHISPER_API void whisper_full_params_set_token_timestamps(struct whisper_full_params *params, bool token_timestamps);
WHISPER_API void whisper_full_params_set_thold_pt(struct whisper_full_params *params, float thold_pt);
WHISPER_API void whisper_full_params_set_thold_ptsum(struct whisper_full_params *params, float thold_ptsum);
WHISPER_API void whisper_full_params_set_max_len(struct whisper_full_params *params, int max_len);
WHISPER_API void whisper_full_params_set_split_on_word(struct whisper_full_params *params, bool split_on_word);
WHISPER_API void whisper_full_params_set_max_tokens(struct whisper_full_params *params, int max_tokens);

WHISPER_API void whisper_full_params_set_speed_up(struct whisper_full_params *params, bool speed_up);
WHISPER_API void whisper_full_params_set_debug_mode(struct whisper_full_params *params, bool debug_mode);
WHISPER_API void whisper_full_params_set_audio_ctx(struct whisper_full_params *params, int audio_ctx);

WHISPER_API void whisper_full_params_set_tdrz_enable(struct whisper_full_params *params, bool tdrz_enable);

WHISPER_API void whisper_full_params_set_initial_prompt(struct whisper_full_params *params, const char *initial_prompt);
WHISPER_API void
whisper_full_params_set_prompt_tokens(struct whisper_full_params *params, const int32_t *prompt_tokens);
WHISPER_API void whisper_full_params_set_prompt_n_tokens(struct whisper_full_params *params, int prompt_n_tokens);

WHISPER_API void whisper_full_params_set_language(struct whisper_full_params *params, const char *language);
WHISPER_API void whisper_full_params_set_detect_language(struct whisper_full_params *params, bool detect_language);

WHISPER_API void whisper_full_params_set_suppress_blank(struct whisper_full_params *params, bool suppress_blank);
WHISPER_API void
whisper_full_params_set_suppress_non_speech_tokens(struct whisper_full_params *params, bool suppress_non_speech_tokens);

WHISPER_API void whisper_full_params_set_temperature(struct whisper_full_params *params, float temperature);
WHISPER_API void whisper_full_params_set_max_initial_ts(struct whisper_full_params *params, float max_initial_ts);
WHISPER_API void whisper_full_params_set_length_penalty(struct whisper_full_params *params, float length_penalty);

WHISPER_API void whisper_full_params_set_temperature_inc(struct whisper_full_params *params, float temperature_inc);
WHISPER_API void whisper_full_params_set_entropy_thold(struct whisper_full_params *params, float entropy_thold);
WHISPER_API void whisper_full_params_set_logprob_thold(struct whisper_full_params *params, float logprob_thold);
WHISPER_API void whisper_full_params_set_no_speech_thold(struct whisper_full_params *params, float no_speech_thold);

WHISPER_API void whisper_full_params_set_greedy_best_of(struct whisper_full_params *params, int best_of);

WHISPER_API void whisper_full_params_set_beam_search_beam_size(struct whisper_full_params *params, int beam_size);
WHISPER_API void whisper_full_params_set_beam_search_patience(struct whisper_full_params *params, float patience);

//==============================whisper_context_ref===============================================================

WHISPER_API struct whisper_context *
whisper_init_from_file_with_params_ref(const char *path_model, struct whisper_context_params *params);
WHISPER_API struct whisper_context *
whisper_init_from_buffer_with_params_ref(void *buffer, size_t buffer_size, struct whisper_context_params *params);

WHISPER_API struct whisper_context *
whisper_init_from_file_with_params_ref_no_state(const char *path_model, struct whisper_context_params *params);
WHISPER_API struct whisper_context *
whisper_init_from_buffer_with_params_ref_no_state(void *buffer, size_t buffer_size,
                                                  struct whisper_context_params *params);

//==============================whisper_full_ref===============================================================
WHISPER_API int whisper_full_ref(
        struct whisper_context *ctx,
        struct whisper_full_params *params,
        const float *samples,
        int n_samples);

WHISPER_API int whisper_full_ref_with_state(
        struct whisper_context *ctx,
        struct whisper_state *state,
        struct whisper_full_params *params,
        const float *samples,
        int n_samples);

WHISPER_API int whisper_full_ref_parallel(
        struct whisper_context *ctx,
        struct whisper_full_params *params,
        const float *samples,
        int n_samples,
        int n_processors);
#ifdef __cplusplus
}
#endif

#endif //WHISPER_ABI_H
