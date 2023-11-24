// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package whisper

import (
	"github.com/ebitengine/purego"
	"unsafe"
)

const (
	WHISPER_SAMPLE_RATE int = 16000
	WHISPER_N_FFT           = 400
	WHISPER_HOP_LENGTH      = 160
	WHISPER_CHUNK_SIZE      = 30
)

const (
	cWhisperInitFromFileWithParams   = "whisper_init_from_file_with_params"
	cWhisperInitFromBufferWithParams = "whisper_init_from_buffer_with_params"
	cWhisperInitWithParams           = "whisper_init_with_params"

	cWhisperInitFromFileWithParamsNoState   = "whisper_init_from_file_with_params_no_state"
	cWhisperInitFromBufferWithParamsNoState = "whisper_init_from_buffer_with_params_no_state"
	cWhisperInitWithParamsNoState           = "whisper_init_with_params_no_state"

	cWhisperInitState              = "whisper_init_state"
	cWhisperCtxInitOpenvinoEncoder = "whisper_ctx_init_openvino_encoder"

	cWhisperFree              = "whisper_free"
	cWhisperFreeState         = "whisper_free_state"
	cWhisperFreeParams        = "whisper_free_params"
	cWhisperFreeContextParams = "whisper_free_context_params"

	cWhisperPcmToMel          = "whisper_pcm_to_mel"
	cWhisperPcmToMelWithState = "whisper_pcm_to_mel_with_state"

	cWhisperPcmToMelPhaseVocoder          = "whisper_pcm_to_mel_phase_vocoder"
	cWhisperPcmToMelPhaseVocoderWithState = "whisper_pcm_to_mel_phase_vocoder_with_state"

	cWhisperSetMel          = "whisper_set_mel"
	cWhisperSetMelWithState = "whisper_set_mel_with_state"

	cWhisperEncode          = "whisper_encode"
	cWhisperEncodeWithState = "whisper_encode_with_state"

	cWhisperDecode          = "whisper_decode"
	cWhisperDecodeWithState = "whisper_decode_with_state"

	cWhisperTokenize = "whisper_tokenize"

	cWhisperLangMaxId = "whisper_lang_max_id"
	cWhisperLangId    = "whisper_lang_id"
	cWhisperLangStr   = "whisper_lang_str"

	cWhisperLangAutoDetect          = "whisper_lang_auto_detect"
	cWhisperLangAutoDetectWithState = "whisper_lang_auto_detect_with_state"

	cWhisperNLen           = "whisper_n_len_from_state"
	cWhisperNLenFromState  = "whisper_n_len_from_state"
	cWhisperNVocab         = "whisper_n_vocab"
	cWhisperNTextCtx       = "whisper_n_text_ctx"
	cWhisperNAudioCtx      = "whisper_n_audio_ctx"
	cWhisperIsMultilingual = "whisper_is_multilingual"

	cWhisperModelNVocab      = "whisper_model_n_vocab"
	cWhisperModelNAudioCtx   = "whisper_model_n_audio_ctx"
	cWhisperModelNAudioState = "whisper_model_n_audio_state"
	cWhisperModelNAudioHead  = "whisper_model_n_audio_head"
	cWhisperModelNAudioLayer = "whisper_model_n_audio_layer"
	cWhisperModelNTextCtx    = "whisper_model_n_text_ctx"
	cWhisperModelNTextState  = "whisper_model_n_text_state"
	cWhisperModelNTextHead   = "whisper_model_n_text_head"
	cWhisperModelNTextLayer  = "whisper_model_n_text_layer"
	cWhisperModelNMels       = "whisper_model_n_mels"
	cWhisperModelFtype       = "whisper_model_ftype"
	cWhisperModelType        = "whisper_model_type"

	cWhisperGetLogits          = "whisper_get_logits"
	cWhisperGetLogitsFromState = "whisper_get_logits_from_state"

	cWhisperTokenToStr        = "whisper_token_to_str"
	cWhisperModelTypeReadable = "whisper_model_type_readable"

	cWhisperTokenEot  = "whisper_token_eot"
	cWhisperTokenSot  = "whisper_token_sot"
	cWhisperTokenSolm = "whisper_token_solm"
	cWhisperTokenPrev = "whisper_token_prev"
	cWhisperTokenNosp = "whisper_token_nosp"
	cWhisperTokenNot  = "whisper_token_not"
	cWhisperTokenBeg  = "whisper_token_beg"
	cWhisperTokenLang = "whisper_token_lang"

	cWhisperTokenTranslate  = "whisper_token_translate"
	cWhisperTokenTranscribe = "whisper_token_transcribe"

	cWhisperPrintTimings = "whisper_print_timings"
	cWhisperResetTimings = "whisper_reset_timings"

	cWhisperPrintSystemInfo = "whisper_print_system_info"

	cWhisperContextDefaultParamsByRef = "whisper_context_default_params_by_ref"
	cWhisperContextDefaultParams      = "whisper_context_default_params"
	cWhisperFullDefaultParamsByRef    = "whisper_full_default_params_by_ref"
	cWhisperFullDefaultParams         = "whisper_full_default_params"

	cWhisperFull          = "whisper_full"
	cWhisperFullWithState = "whisper_full_with_state"

	cWhisperFullParallel  = "whisper_full_parallel"
	cWhisperFullNSegments = "whisper_full_n_segments"

	cWhisperFullNSegmentsFromState = "whisper_full_n_segments_from_state"

	cWhisperFullLangId = "whisper_full_lang_id"

	cWhisperFullLangIdFromState = "whisper_full_lang_id_from_state"

	cWhisperFullGetSegmentT0          = "whisper_full_get_segment_t0"
	cWhisperFullGetSegmentT0FromState = "whisper_full_get_segment_t0_from_state"

	cWhisperFullGetSegmentT1          = "whisper_full_get_segment_t1"
	cWhisperFullGetSegmentT1FromState = "whisper_full_get_segment_t1_from_state"

	cWhisperFullGetSegmentSpeakerTurnNext          = "whisper_full_get_segment_speaker_turn_next"
	cWhisperFullGetSegmentSpeakerTurnNextFromState = "whisper_full_get_segment_speaker_turn_next_from_state"

	cWhisperFullGetSegmentText          = "whisper_full_get_segment_text"
	cWhisperFullGetSegmentTextFromState = "whisper_full_get_segment_text_from_state"

	cWhisperFullNTokens          = "whisper_full_n_tokens"
	cWhisperFullNTokensFromState = "whisper_full_n_tokens_from_state"

	cWhisperFullGetTokenText          = "whisper_full_get_token_text"
	cWhisperFullGetTokenTextFromState = "whisper_full_get_token_text_from_state"

	cWhisperFullGetTokenId          = "whisper_full_get_token_id"
	cWhisperFullGetTokenIdFromState = "whisper_full_get_token_id_from_state"

	cWhisperFullGetTokenData          = "whisper_full_get_token_data"
	cWhisperFullGetTokenDataFromState = "whisper_full_get_token_data_from_state"

	cWhisperFullGetTokenP          = "whisper_full_get_token_p"
	cWhisperFullGetTokenPFromState = "whisper_full_get_token_p_from_state"
)

type WhisperContext struct {
	ctx uintptr
}

type WhisperState struct {
	state uintptr
}

// WhisperFullParams Parameters for the CWhisper.WhisperFull function
// If you change the order or add new parameters, make sure to update the default values in whisper.cpp:
// CWhisper.WhisperFullDefaultParams()
// TODO: align memo
type WhisperFullParams struct {
	strategy int32

	nThreads    int32
	nMaxTextCtx int32 // max tokens to use from past text as prompt for the decoder
	offsetMs    int32 // start offset in ms
	durationMs  int32 // audio duration to process in ms

	translate       bool
	noContext       bool // do not use past transcription (if any) as initial prompt for the decoder
	noTimestamps    bool // do not generate timestamps
	singleSegment   bool // force single segment output (useful for streaming)
	printSpecial    bool // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
	printProgress   bool // print progress information
	printRealtime   bool // print results from within whisper.cpp (avoid it, use callback instead)
	printTimestamps bool // print timestamps for each text segment when printing realtime

	// [EXPERIMENTAL] token-level timestamps
	tokenTimestamps bool    // enable token-level timestamps
	tholdPt         float32 // timestamp token probability threshold (~0.01)
	tholdPtsum      float32 // timestamp token sum probability threshold (~0.01)
	maxLen          int32   // max segment length in characters
	splitOnWord     bool    // split on word rather than on token (when used with max_len)
	maxTokens       int32   // max tokens per segment (0 = no limit)

	// [EXPERIMENTAL] speed-up techniques
	// note: these can significantly reduce the quality of the output
	speedUp   bool  // speed-up the audio by 2x using Phase Vocoder
	debugMode bool  // enable debug_mode provides extra info (eg. Dump log_mel)
	audioCtx  int32 // overwrite the audio context size (0 = use default)

	// [EXPERIMENTAL] [TDRZ] tinydiarize
	tdrzEnable bool // enable tinydiarize speaker turn detection

	// tokens to provide to the whisper decoder as initial prompt
	// these are prepended to any existing text context from a previous call
	initialPrompt uintptr
	promptTokens  uintptr
	promptNTokens int32

	// for auto-detection, set to nullptr, "" or "auto"
	language       uintptr
	detectLanguage bool

	// common decoding parameters:
	suppressBlank           bool // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
	suppressNonSpeechTokens bool // ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253

	temperature   float32 // initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
	maxInitialTs  float32 // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
	lengthPenalty float32 // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267

	// fallback parameters
	// ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
	temperatureInc float32
	entropyThold   float32 // similar to OpenAI's "compression_ratio_threshold"
	logprobThold   float32
	noSpeechThold  float32 // [whisper.cpp] TODO: not implemented

	//struct {
	//	int best_of;    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L264
	//} greedy;
	greedy uintptr

	//struct {
	//	int beam_size;  // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L265
	//
	//	float patience; // TODO: not implemented, ref: https://arxiv.org/pdf/2204.05424.pdf
	//} beam_search;
	beamSearch uintptr

	// called for every newly generated text segment
	newSegmentCallback         uintptr
	newSegmentCallbackUserData uintptr

	// called on each progress update
	progressCallback         uintptr
	progressCallbackUserData uintptr

	// called each time before the encoder starts
	encoderBeginCallback         uintptr
	encoderBeginCallbackUserData uintptr

	// called each time before ggml computation starts
	abortCallback         uintptr
	abortCallbackUserData uintptr

	// called by each decoder to filter obtained logits
	logitsFilterCallback         uintptr
	logitsFilterCallbackUserData uintptr

	grammarRules   uintptr
	nGrammarRules  int32
	iStartRule     int32
	grammarPenalty float32
}

type WhisperFullParamsRef struct {
	paramsRef uintptr
}

type WhisperContextParams struct {
	UseGpu bool
}

func (p *WhisperContextParams) ToWhisperContextParamsRef() *WhisperContextParamsRef {
	return &WhisperContextParamsRef{paramsRef: uintptr(unsafe.Pointer(p))}
}

type WhisperContextParamsRef struct {
	paramsRef uintptr
}

func (r *WhisperContextParamsRef) ToWhisperFullParams() *WhisperContextParams {
	return (*WhisperContextParams)(unsafe.Pointer(r.paramsRef))
}

type WhisperPos int
type WhisperToken int
type WhisperSeqId int

type WhisperSamplingStrategy int32

const (
	WHISPER_SAMPLING_GREEDY WhisperSamplingStrategy = iota
	WHISPER_SAMPLING_BEAM_SEARCH
)

type WhisperTokenData struct {
	// todo
}

type CWhisper interface {
	// WhisperInitFromFileWithParams WhisperInitFromBufferWithParams Various functions for loading a ggml whisper model.
	// Allocate (almost) all memory needed for the model.
	// Return NULL on failure
	WhisperInitFromFileWithParams(pathModel string, params *WhisperContextParamsRef) *WhisperContext
	WhisperInitFromBufferWithParams(buffer []byte, params *WhisperContextParamsRef) *WhisperContext
	// whisper_init_with_params(struct whisper_model_loader * loader, struct whisper_context_params params) error

	// WhisperInitFromFileWithParamsNoState These are the same as the above, but the internal state of the context is not allocated automatically
	// It is the responsibility of the caller to allocate the state using whisper_init_state() (#523)
	WhisperInitFromFileWithParamsNoState(pathModel string, params *WhisperContextParamsRef) *WhisperContext
	WhisperInitFromBufferWithParamsNoState(buffer []byte, params *WhisperContextParamsRef) *WhisperContext
	//whisper_init_with_params_no_state(struct whisper_model_loader * loader, struct whisper_context_params params)

	WhisperInitState(ctx *WhisperContext) *WhisperState

	// WhisperCtxInitOpenvinoEncoder Given a context, enable use of OpenVINO for encode inference.
	// model_path: Optional path to OpenVINO encoder IR model. If set to nullptr,
	//                      the path will be generated from the ggml model path that was passed
	//                      in to whisper_init_from_file. For example, if 'path_model' was
	//                      "/path/to/ggml-base.en.bin", then OpenVINO IR model path will be
	//                      assumed to be "/path/to/ggml-base.en-encoder-openvino.xml".
	// device: OpenVINO device to run inference on ("CPU", "GPU", etc.)
	// cache_dir: Optional cache directory that can speed up init time, especially for
	//                     GPU, by caching compiled 'blobs' there.
	//                     Set to nullptr if not used.
	// Returns 0 on success. If OpenVINO is not enabled in build, this simply returns 1.
	WhisperCtxInitOpenvinoEncoder(ctx *WhisperContext, modelPath, device, cacheDir string) *WhisperState

	// WhisperFree WhisperFreeState WhisperFreeParams WhisperFreeContextParams Frees all allocated memory
	WhisperFree(ctx *WhisperContext)
	WhisperFreeState(state *WhisperState)
	WhisperFreeParams(params *WhisperFullParamsRef)
	WhisperFreeContextParams(params *WhisperContextParamsRef)

	// WhisperPcmToMel Convert RAW PCM audio to log mel spectrogram.
	// The resulting spectrogram is stored inside the default state of the provided whisper context.
	// Returns 0 on success
	WhisperPcmToMel(ctx *WhisperContext, samples []float32, nSamples, nThreads int) int
	WhisperPcmToMelWithState(ctx *WhisperContext, state *WhisperState, samples []float32, nSamples, nThreads int) int

	// WhisperPcmToMelPhaseVocoder Convert RAW PCM audio to log mel spectrogram but applies a Phase Vocoder to speed up the audio x2.
	// The resulting spectrogram is stored inside the default state of the provided whisper context.
	// Returns 0 on success
	WhisperPcmToMelPhaseVocoder(ctx *WhisperContext, samples []float32, nSamples, nThreads int) int
	WhisperPcmToMelPhaseVocoderWithState(ctx *WhisperContext, state *WhisperState, samples []float32, nSamples, nThreads int) int

	// WhisperSetMel This can be used to set a custom log mel spectrogram inside the default state of the provided whisper context.
	// Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
	// n_mel must be 80
	// Returns 0 on success
	WhisperSetMel(ctx *WhisperContext, data []float32, nLen, nMel int) int
	WhisperSetMelWithState(ctx *WhisperContext, state *WhisperState, data []float32, nLen, nMel int) int

	// WhisperEncode Run the Whisper encoder on the log mel spectrogram stored inside the default state in the provided whisper context.
	// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
	// offset can be used to specify the offset of the first frame in the spectrogram.
	// Returns 0 on success
	WhisperEncode(ctx *WhisperContext, offset, nThreads int) int
	WhisperEncodeWithState(ctx *WhisperContext, state *WhisperState, offset, nThreads int) int

	// WhisperDecode Run the Whisper decoder to obtain the logits and probabilities for the next token.
	// Make sure to call whisper_encode() first.
	// tokens + n_tokens is the provided context for the decoder.
	// n_past is the number of tokens to use from previous decoder calls.
	// Returns 0 on success
	// [whisper.cpp] TODO: add support for multiple decoders
	WhisperDecode(ctx *WhisperContext, tokens []WhisperToken, nTokens, nPast, nThreads int) int
	WhisperDecodeWithState(ctx *WhisperContext, state *WhisperState, tokens []WhisperToken, nTokens, nPast, nThreads int) int

	// WhisperTokenize Convert the provided text into tokens.
	// The tokens pointer must be large enough to hold the resulting tokens.
	// Returns the number of tokens on success, no more than n_max_tokens
	// Returns -1 on failure
	// [whisper.cpp] TODO: not sure if correct
	WhisperTokenize(ctx *WhisperContext, text string, tokens []WhisperToken, nMaxTokens int) int

	// WhisperLangMaxId Largest language id (i.e. number of available languages - 1)
	WhisperLangMaxId() int

	// WhisperLangId Return the id of the specified language, returns -1 if not found
	// Examples:
	//   "de" -> 2
	//   "german" -> 2
	WhisperLangId(lang string) int

	// WhisperLangStr Return the short string of the specified language id (e.g. 2 -> "de"), returns nullptr if not found
	WhisperLangStr(id int) string

	// WhisperLangAutoDetect Use mel data at offset_ms to try and auto-detect the spoken language
	// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first
	// Returns the top language id or negative on failure
	// If not null, fills the lang_probs array with the probabilities of all languages
	// The array must be whisper_lang_max_id() + 1 in size
	// ref: https://github.com/openai/whisper/blob/main/whisper/decoding.py#L18-L69
	WhisperLangAutoDetect(ctx *WhisperContext, offsetMs, nThreads int, langProbs []float32) int
	WhisperLangAutoDetectWithState(ctx *WhisperContext, state *WhisperState, offsetMs, nThreads int, langProbs []float32) int

	// WhisperNLen mel length
	WhisperNLen(ctx *WhisperContext) int
	// WhisperNLenFromState mel length
	WhisperNLenFromState(state *WhisperState) int
	WhisperNVocab(ctx *WhisperContext) int
	WhisperNTextCtx(ctx *WhisperContext) int
	WhisperNAudioCtx(ctx *WhisperContext) int
	WhisperIsMultilingual(ctx *WhisperContext) int

	WhisperModelNVocab(ctx *WhisperContext) int
	WhisperModelNAudioCtx(ctx *WhisperContext) int
	WhisperModelNAudioState(ctx *WhisperContext) int
	WhisperModelNAudioHead(ctx *WhisperContext) int
	WhisperModelNAudioLayer(ctx *WhisperContext) int
	WhisperModelNTextCtx(ctx *WhisperContext) int
	WhisperModelNTextState(ctx *WhisperContext) int
	WhisperModelNTextHead(ctx *WhisperContext) int
	WhisperModelNTextLayer(ctx *WhisperContext) int
	WhisperModelNMels(ctx *WhisperContext) int
	WhisperModelFtype(ctx *WhisperContext) int
	WhisperModelType(ctx *WhisperContext) int

	// WhisperGetLogits Token logits obtained from the last call to whisper_decode()
	// The logits for the last token are stored in the last row
	// Rows: n_tokens
	// Cols: n_vocab
	WhisperGetLogits(ctx *WhisperContext) []float32
	WhisperGetLogitsFromState(state *WhisperState) []float32

	// WhisperTokenToStr Token Id -> String. Uses the vocabulary in the provided context
	WhisperTokenToStr(ctx *WhisperContext, token WhisperToken) string
	WhisperModelTypeReadable(ctx *WhisperContext) string

	// WhisperTokenEot WhisperTokenSot WhisperTokenSolm WhisperTokenPrev WhisperTokenNosp WhisperTokenNot WhisperTokenBeg WhisperTokenLang Special tokens
	WhisperTokenEot(ctx *WhisperContext) WhisperToken
	WhisperTokenSot(ctx *WhisperContext) WhisperToken
	WhisperTokenSolm(ctx *WhisperContext) WhisperToken
	WhisperTokenPrev(ctx *WhisperContext) WhisperToken
	WhisperTokenNosp(ctx *WhisperContext) WhisperToken
	WhisperTokenNot(ctx *WhisperContext) WhisperToken
	WhisperTokenBeg(ctx *WhisperContext) WhisperToken
	WhisperTokenLang(ctx *WhisperContext) WhisperToken

	// WhisperTokenTranslate WhisperTokenTranscribe Task tokens
	WhisperTokenTranslate(ctx *WhisperContext) WhisperToken
	WhisperTokenTranscribe(ctx *WhisperContext) WhisperToken

	// WhisperPrintTimings WhisperResetTimings Performance information from the default state.
	WhisperPrintTimings(ctx *WhisperContext)
	WhisperResetTimings(ctx *WhisperContext)

	// WhisperPrintSystemInfo Print system information
	WhisperPrintSystemInfo() string

	// WhisperContextDefaultParamsByRef NOTE: this function allocates memory, and it is the responsibility of the caller to free the pointer - see whisper_free_context_params & whisper_free_params()
	WhisperContextDefaultParamsByRef() *WhisperContextParamsRef
	WhisperContextDefaultParams() *WhisperContextParams
	WhisperFullDefaultParamsByRef(strategy WhisperSamplingStrategy) *WhisperFullParamsRef
	WhisperFullDefaultParams(strategy WhisperSamplingStrategy) *WhisperFullParams

	// WhisperFull Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
	// Not thread safe for same context
	// Uses the specified decoding strategy to obtain the text.
	WhisperFull(ctx *WhisperContext, params *WhisperFullParamsRef, samples []float32, nSamples int) int

	WhisperFullWithState(ctx *WhisperContext, state *WhisperState, params *WhisperFullParamsRef, samples []float32, nSamples int) int

	// WhisperFullParallel Split the input audio in chunks and process each chunk separately using whisper_full_with_state()
	// Result is stored in the default state of the context
	// Not thread safe if executed in parallel on the same context.
	// It seems this approach can offer some speedup in some cases.
	// However, the transcription accuracy can be worse at the beginning and end of each chunk.
	WhisperFullParallel(ctx *WhisperContext, params *WhisperFullParamsRef, samples []float32, nSamples, nProcessors int) int

	// WhisperFullNSegments WhisperFullNSegmentsFromState Number of generated text segments
	// A segment can be a few words, a sentence, or even a paragraph.
	WhisperFullNSegments(ctx *WhisperContext) int
	WhisperFullNSegmentsFromState(state *WhisperState) int

	// WhisperFullLangId Language id associated with the context's default state
	WhisperFullLangId(ctx *WhisperContext) int

	// WhisperFullLangIdFromState Language id associated with the provided state
	WhisperFullLangIdFromState(state *WhisperState) int

	// WhisperFullGetSegmentT0 WhisperFullGetSegmentT0FromState Get the start and end time of the specified segment
	WhisperFullGetSegmentT0(ctx *WhisperContext, iSegment int) int64
	WhisperFullGetSegmentT0FromState(state *WhisperState, iSegment int) int64

	WhisperFullGetSegmentT1(ctx *WhisperContext, iSegment int) int64
	WhisperFullGetSegmentT1FromState(state *WhisperState, iSegment int) int64

	// WhisperFullGetSegmentSpeakerTurnNext WhisperFullGetSegmentSpeakerTurnNextFromState Get whether the next segment is predicted as a speaker turn
	WhisperFullGetSegmentSpeakerTurnNext(ctx *WhisperContext, iSegment int) bool
	WhisperFullGetSegmentSpeakerTurnNextFromState(state *WhisperState, iSegment int) bool

	// WhisperFullGetSegmentText WhisperFullGetSegmentTextFromState Get the text of the specified segment
	WhisperFullGetSegmentText(ctx *WhisperContext, iSegment int) string
	WhisperFullGetSegmentTextFromState(state *WhisperState, iSegment int) string

	// WhisperFullNTokens Get number of tokens in the specified segment
	WhisperFullNTokens(ctx *WhisperContext, iSegment int) int
	WhisperFullNTokensFromState(state *WhisperState, iSegment int) int

	// WhisperFullGetTokenText Get the token text of the specified token in the specified segment
	WhisperFullGetTokenText(ctx *WhisperContext, iSegment, iToken int) string
	WhisperFullGetTokenTextFromState(ctx *WhisperContext, state *WhisperState, iSegment, iToken int) string

	WhisperFullGetTokenId(ctx *WhisperContext, iSegment, iToken int) WhisperToken
	WhisperFullGetTokenIdFromState(state *WhisperState, iSegment, iToken int) WhisperToken

	// WhisperFullGetTokenData WhisperFullGetTokenDataFromState Get token data for the specified token in the specified segment
	// This contains probabilities, timestamps, etc.
	WhisperFullGetTokenData(ctx *WhisperContext, iSegment, iToken int) WhisperTokenData
	WhisperFullGetTokenDataFromState(state *WhisperState, iSegment, iToken int) WhisperTokenData

	// WhisperFullGetTokenP WhisperFullGetTokenPFromState Get the probability of the specified token in the specified segment
	WhisperFullGetTokenP(ctx *WhisperContext, iSegment, iToken int) float32
	WhisperFullGetTokenPFromState(state *WhisperState, iSegment, iToken int) float32
}

type CWhisperImpl struct {
	libWhisper                       uintptr
	cWhisperInitFromFileWithParams   func(pathModel string, params uintptr) uintptr
	cWhisperInitFromBufferWithParams func(buffer uintptr, bufferSize int, params uintptr) uintptr
	//cWhisperInitWithParams           func()

	cWhisperInitFromFileWithParamsNoState   func(pathModel string, params uintptr) uintptr
	cWhisperInitFromBufferWithParamsNoState func(buffer uintptr, bufferSize int, params uintptr) uintptr
	//cWhisperInitWithParamsNoState           func()

	cWhisperInitState              func(ctx uintptr) uintptr
	cWhisperCtxInitOpenvinoEncoder func(ctx uintptr, modelPath, device, cacheDir string) uintptr

	cWhisperFree              func(ctx uintptr)
	cWhisperFreeState         func(state uintptr)
	cWhisperFreeParams        func(params uintptr)
	cWhisperFreeContextParams func(params uintptr)

	cWhisperPcmToMel          func(ctx uintptr, samples *float32, nSamples, nThreads int) int
	cWhisperPcmToMelWithState func(ctx uintptr, state uintptr, samples *float32, nSamples, nThreads int) int

	cWhisperPcmToMelPhaseVocoder          func(ctx uintptr, samples *float32, nSamples, nThreads int) int
	cWhisperPcmToMelPhaseVocoderWithState func(ctx uintptr, state uintptr, samples *float32, nSamples, nThreads int) int

	cWhisperSetMel          func(ctx uintptr, data *float32, nLen, nMel int) int
	cWhisperSetMelWithState func(ctx uintptr, state uintptr, data *float32, nLen, nMel int) int

	cWhisperEncode          func(ctx uintptr, offset, nThreads int) int
	cWhisperEncodeWithState func(ctx uintptr, state uintptr, offset, nThreads int) int

	cWhisperDecode          func(ctx uintptr, tokens *int, nTokens, nPast, nThreads int) int
	cWhisperDecodeWithState func(ctx uintptr, state uintptr, tokens *int, nTokens, nPast, nThreads int) int

	cWhisperTokenize func(ctx uintptr, text string, tokens *int, nMaxTokens int) int

	cWhisperLangMaxId func() int
	cWhisperLangId    func(lang string) int
	cWhisperLangStr   func(lang int) string

	cWhisperLangAutoDetect          func(ctx uintptr, offsetMs, nThreads int, langProbs *float32) int
	cWhisperLangAutoDetectWithState func(ctx uintptr, state uintptr, offsetMs, nThreads int, langProbs *float32) int

	cWhisperNLen           func(ctx uintptr) int
	cWhisperNLenFromState  func(ctx uintptr) int
	cWhisperNVocab         func(ctx uintptr) int
	cWhisperNTextCtx       func(ctx uintptr) int
	cWhisperNAudioCtx      func(ctx uintptr) int
	cWhisperIsMultilingual func(ctx uintptr) int

	cWhisperModelNVocab      func(ctx uintptr) int
	cWhisperModelNAudioCtx   func(ctx uintptr) int
	cWhisperModelNAudioState func(ctx uintptr) int
	cWhisperModelNAudioHead  func(ctx uintptr) int
	cWhisperModelNAudioLayer func(ctx uintptr) int
	cWhisperModelNTextCtx    func(ctx uintptr) int
	cWhisperModelNTextState  func(ctx uintptr) int
	cWhisperModelNTextHead   func(ctx uintptr) int
	cWhisperModelNTextLayer  func(ctx uintptr) int
	cWhisperModelNMels       func(ctx uintptr) int
	cWhisperModelFtype       func(ctx uintptr) int
	cWhisperModelType        func(ctx uintptr) int

	cWhisperGetLogits          func(ctx uintptr) *float32
	cWhisperGetLogitsFromState func(state uintptr) *float32

	cWhisperTokenToStr        func(ctx uintptr, token int) string
	cWhisperModelTypeReadable func(ctx uintptr) string

	cWhisperTokenEot  func(ctx uintptr) int
	cWhisperTokenSot  func(ctx uintptr) int
	cWhisperTokenSolm func(ctx uintptr) int
	cWhisperTokenPrev func(ctx uintptr) int
	cWhisperTokenNosp func(ctx uintptr) int
	cWhisperTokenNot  func(ctx uintptr) int
	cWhisperTokenBeg  func(ctx uintptr) int
	cWhisperTokenLang func(ctx uintptr) int

	cWhisperTokenTranslate  func(ctx uintptr) int
	cWhisperTokenTranscribe func(ctx uintptr) int

	cWhisperPrintTimings func(ctx uintptr)
	cWhisperResetTimings func(ctx uintptr)

	cWhisperPrintSystemInfo func() string

	cWhisperContextDefaultParamsByRef func() uintptr
	cWhisperContextDefaultParams      func() uintptr
	cWhisperFullDefaultParamsByRef    func(strategy int) uintptr
	cWhisperFullDefaultParams         func(strategy int) uintptr

	cWhisperFull          func(ctx uintptr, params uintptr, samples *float32, nSamples int) int
	cWhisperFullWithState func(ctx uintptr, state uintptr, params uintptr, samples *float32, nSamples int) int

	cWhisperFullParallel func(ctx uintptr, params uintptr, samples *float32, nSamples, nProcessors int) int

	cWhisperFullNSegments          func(ctx uintptr) int
	cWhisperFullNSegmentsFromState func(state uintptr) int

	cWhisperFullLangId func(ctx uintptr) int

	cWhisperFullLangIdFromState func(state uintptr) int

	cWhisperFullGetSegmentT0          func(ctx uintptr, iSegment int) int64
	cWhisperFullGetSegmentT0FromState func(state uintptr, iSegment int) int64

	cWhisperFullGetSegmentT1          func(ctx uintptr, iSegment int) int64
	cWhisperFullGetSegmentT1FromState func(state uintptr, iSegment int) int64

	cWhisperFullGetSegmentSpeakerTurnNext          func(ctx uintptr, iSegment int) bool
	cWhisperFullGetSegmentSpeakerTurnNextFromState func(state uintptr, iSegment int) bool

	cWhisperFullGetSegmentText          func(ctx uintptr, iSegment int) string
	cWhisperFullGetSegmentTextFromState func(ctx uintptr, state uintptr, iSegment int) string

	cWhisperFullNTokens          func(ctx uintptr, iSegment int) int
	cWhisperFullNTokensFromState func(state uintptr, iSegment int) int

	cWhisperFullGetTokenText          func(ctx uintptr, iSegment, iToken int) string
	cWhisperFullGetTokenTextFromState func(ctx uintptr, state uintptr, iSegment, iToken int) string

	cWhisperFullGetTokenId          func(ctx uintptr, iSegment, iToken int) int
	cWhisperFullGetTokenIdFromState func(state uintptr, iSegment, iToken int) int

	//TODO
	//cWhisperFullGetTokenData          func()
	//cWhisperFullGetTokenDataFromState func()

	cWhisperFullGetTokenP          func(ctx uintptr, iSegment, iToken int) float32
	cWhisperFullGetTokenPFromState func(state uintptr, iSegment, iToken int) float32
}

func NewCWhisper(libraryPath string) (*CWhisperImpl, error) {
	libWhisper, err := openLibrary(libraryPath)
	if err != nil {
		return nil, err
	}
	var (
		whisperInitFromFileWithParams   func(pathModel string, params uintptr) uintptr
		whisperInitFromBufferWithParams func(buffer uintptr, bufferSize int, params uintptr) uintptr
		//whisperInitWithParams           func()

		whisperInitFromFileWithParamsNoState   func(pathModel string, params uintptr) uintptr
		whisperInitFromBufferWithParamsNoState func(buffer uintptr, bufferSize int, params uintptr) uintptr
		//whisperInitWithParamsNoState           func()

		whisperInitState              func(ctx uintptr) uintptr
		whisperCtxInitOpenvinoEncoder func(ctx uintptr, modelPath, device, cacheDir string) uintptr

		whisperFree              func(ctx uintptr)
		whisperFreeState         func(state uintptr)
		whisperFreeParams        func(params uintptr)
		whisperFreeContextParams func(params uintptr)

		whisperPcmToMel          func(ctx uintptr, samples *float32, nSamples, nThreads int) int
		whisperPcmToMelWithState func(ctx uintptr, state uintptr, samples *float32, nSamples, nThreads int) int

		whisperPcmToMelPhaseVocoder          func(ctx uintptr, samples *float32, nSamples, nThreads int) int
		whisperPcmToMelPhaseVocoderWithState func(ctx uintptr, state uintptr, samples *float32, nSamples, nThreads int) int

		whisperSetMel          func(ctx uintptr, data *float32, nLen, nMel int) int
		whisperSetMelWithState func(ctx uintptr, state uintptr, data *float32, nLen, nMel int) int

		whisperEncode          func(ctx uintptr, offset, nThreads int) int
		whisperEncodeWithState func(ctx uintptr, state uintptr, offset, nThreads int) int

		whisperDecode          func(ctx uintptr, tokens *int, nTokens, nPast, nThreads int) int
		whisperDecodeWithState func(ctx uintptr, state uintptr, tokens *int, nTokens, nPast, nThreads int) int

		whisperTokenize func(ctx uintptr, text string, tokens *int, nMaxTokens int) int

		whisperLangMaxId func() int
		whisperLangId    func(lang string) int
		whisperLangStr   func(lang int) string

		whisperLangAutoDetect          func(ctx uintptr, offsetMs, nThreads int, langProbs *float32) int
		whisperLangAutoDetectWithState func(ctx uintptr, state uintptr, offsetMs, nThreads int, langProbs *float32) int

		whisperNLen           func(ctx uintptr) int
		whisperNLenFromState  func(ctx uintptr) int
		whisperNVocab         func(ctx uintptr) int
		whisperNTextCtx       func(ctx uintptr) int
		whisperNAudioCtx      func(ctx uintptr) int
		whisperIsMultilingual func(ctx uintptr) int

		whisperModelNVocab      func(ctx uintptr) int
		whisperModelNAudioCtx   func(ctx uintptr) int
		whisperModelNAudioState func(ctx uintptr) int
		whisperModelNAudioHead  func(ctx uintptr) int
		whisperModelNAudioLayer func(ctx uintptr) int
		whisperModelNTextCtx    func(ctx uintptr) int
		whisperModelNTextState  func(ctx uintptr) int
		whisperModelNTextHead   func(ctx uintptr) int
		whisperModelNTextLayer  func(ctx uintptr) int
		whisperModelNMels       func(ctx uintptr) int
		whisperModelFtype       func(ctx uintptr) int
		whisperModelType        func(ctx uintptr) int

		whisperGetLogits          func(ctx uintptr) *float32
		whisperGetLogitsFromState func(state uintptr) *float32

		whisperTokenToStr        func(ctx uintptr, token int) string
		whisperModelTypeReadable func(ctx uintptr) string

		whisperTokenEot  func(ctx uintptr) int
		whisperTokenSot  func(ctx uintptr) int
		whisperTokenSolm func(ctx uintptr) int
		whisperTokenPrev func(ctx uintptr) int
		whisperTokenNosp func(ctx uintptr) int
		whisperTokenNot  func(ctx uintptr) int
		whisperTokenBeg  func(ctx uintptr) int
		whisperTokenLang func(ctx uintptr) int

		whisperTokenTranslate  func(ctx uintptr) int
		whisperTokenTranscribe func(ctx uintptr) int

		whisperPrintTimings func(ctx uintptr)
		whisperResetTimings func(ctx uintptr)

		whisperPrintSystemInfo func() string

		whisperContextDefaultParamsByRef func() uintptr
		whisperContextDefaultParams      func() uintptr
		whisperFullDefaultParamsByRef    func(strategy int) uintptr
		whisperFullDefaultParams         func(strategy int) uintptr

		whisperFull          func(ctx uintptr, params uintptr, samples *float32, nSamples int) int
		whisperFullWithState func(ctx uintptr, state uintptr, params uintptr, samples *float32, nSamples int) int

		whisperFullParallel func(ctx uintptr, params uintptr, samples *float32, nSamples, nProcessors int) int

		whisperFullNSegments          func(ctx uintptr) int
		whisperFullNSegmentsFromState func(state uintptr) int

		whisperFullLangId func(ctx uintptr) int

		whisperFullLangIdFromState func(state uintptr) int

		whisperFullGetSegmentT0          func(ctx uintptr, iSegment int) int64
		whisperFullGetSegmentT0FromState func(state uintptr, iSegment int) int64

		whisperFullGetSegmentT1          func(ctx uintptr, iSegment int) int64
		whisperFullGetSegmentT1FromState func(state uintptr, iSegment int) int64

		whisperFullGetSegmentSpeakerTurnNext          func(ctx uintptr, iSegment int) bool
		whisperFullGetSegmentSpeakerTurnNextFromState func(state uintptr, iSegment int) bool

		whisperFullGetSegmentText          func(ctx uintptr, iSegment int) string
		whisperFullGetSegmentTextFromState func(ctx uintptr, state uintptr, iSegment int) string

		whisperFullNTokens          func(ctx uintptr, iSegment int) int
		whisperFullNTokensFromState func(state uintptr, iSegment int) int

		whisperFullGetTokenText          func(ctx uintptr, iSegment, iToken int) string
		whisperFullGetTokenTextFromState func(ctx uintptr, state uintptr, iSegment, iToken int) string

		whisperFullGetTokenId          func(ctx uintptr, iSegment, iToken int) int
		whisperFullGetTokenIdFromState func(state uintptr, iSegment, iToken int) int

		//whisperFullGetTokenData          func()
		//whisperFullGetTokenDataFromState func()

		whisperFullGetTokenP          func(ctx uintptr, iSegment, iToken int) float32
		whisperFullGetTokenPFromState func(state uintptr, iSegment, iToken int) float32
	)
	purego.RegisterLibFunc(&whisperInitFromFileWithParams, libWhisper, cWhisperInitFromFileWithParams)
	purego.RegisterLibFunc(&whisperInitFromBufferWithParams, libWhisper, cWhisperInitFromBufferWithParams)
	//purego.RegisterLibFunc(&whisperInitWithParams, libWhisper, cWhisperInitWithParams)

	purego.RegisterLibFunc(&whisperInitFromFileWithParamsNoState, libWhisper, cWhisperInitFromFileWithParamsNoState)
	purego.RegisterLibFunc(&whisperInitFromBufferWithParamsNoState, libWhisper, cWhisperInitFromBufferWithParamsNoState)
	//purego.RegisterLibFunc(&whisperInitWithParamsNoState, libWhisper, cWhisperInitWithParamsNoState)

	purego.RegisterLibFunc(&whisperInitState, libWhisper, cWhisperInitState)
	purego.RegisterLibFunc(&whisperCtxInitOpenvinoEncoder, libWhisper, cWhisperCtxInitOpenvinoEncoder)

	purego.RegisterLibFunc(&whisperFree, libWhisper, cWhisperFree)
	purego.RegisterLibFunc(&whisperFreeState, libWhisper, cWhisperFreeState)
	purego.RegisterLibFunc(&whisperFreeParams, libWhisper, cWhisperFreeParams)
	purego.RegisterLibFunc(&whisperFreeContextParams, libWhisper, cWhisperFreeContextParams)

	purego.RegisterLibFunc(&whisperPcmToMel, libWhisper, cWhisperPcmToMel)
	purego.RegisterLibFunc(&whisperPcmToMelWithState, libWhisper, cWhisperPcmToMelWithState)

	purego.RegisterLibFunc(&whisperPcmToMelPhaseVocoder, libWhisper, cWhisperPcmToMelPhaseVocoder)
	purego.RegisterLibFunc(&whisperPcmToMelPhaseVocoderWithState, libWhisper, cWhisperPcmToMelPhaseVocoderWithState)

	purego.RegisterLibFunc(&whisperSetMel, libWhisper, cWhisperSetMel)
	purego.RegisterLibFunc(&whisperSetMelWithState, libWhisper, cWhisperSetMelWithState)

	purego.RegisterLibFunc(&whisperEncode, libWhisper, cWhisperEncode)
	purego.RegisterLibFunc(&whisperEncodeWithState, libWhisper, cWhisperEncodeWithState)

	purego.RegisterLibFunc(&whisperDecode, libWhisper, cWhisperDecode)
	purego.RegisterLibFunc(&whisperDecodeWithState, libWhisper, cWhisperDecodeWithState)

	purego.RegisterLibFunc(&whisperTokenize, libWhisper, cWhisperTokenize)

	purego.RegisterLibFunc(&whisperLangMaxId, libWhisper, cWhisperLangMaxId)
	purego.RegisterLibFunc(&whisperLangId, libWhisper, cWhisperLangId)
	purego.RegisterLibFunc(&whisperLangStr, libWhisper, cWhisperLangStr)

	purego.RegisterLibFunc(&whisperLangAutoDetect, libWhisper, cWhisperLangAutoDetect)
	purego.RegisterLibFunc(&whisperLangAutoDetectWithState, libWhisper, cWhisperLangAutoDetectWithState)

	purego.RegisterLibFunc(&whisperNLen, libWhisper, cWhisperNLen)
	purego.RegisterLibFunc(&whisperNLenFromState, libWhisper, cWhisperNLenFromState)
	purego.RegisterLibFunc(&whisperNVocab, libWhisper, cWhisperNVocab)
	purego.RegisterLibFunc(&whisperNTextCtx, libWhisper, cWhisperNTextCtx)
	purego.RegisterLibFunc(&whisperNAudioCtx, libWhisper, cWhisperNAudioCtx)
	purego.RegisterLibFunc(&whisperIsMultilingual, libWhisper, cWhisperIsMultilingual)

	purego.RegisterLibFunc(&whisperModelNVocab, libWhisper, cWhisperModelNVocab)
	purego.RegisterLibFunc(&whisperModelNAudioCtx, libWhisper, cWhisperModelNAudioCtx)
	purego.RegisterLibFunc(&whisperModelNAudioState, libWhisper, cWhisperModelNAudioState)
	purego.RegisterLibFunc(&whisperModelNAudioHead, libWhisper, cWhisperModelNAudioHead)
	purego.RegisterLibFunc(&whisperModelNAudioLayer, libWhisper, cWhisperModelNAudioLayer)
	purego.RegisterLibFunc(&whisperModelNTextCtx, libWhisper, cWhisperModelNTextCtx)
	purego.RegisterLibFunc(&whisperModelNTextState, libWhisper, cWhisperModelNTextState)
	purego.RegisterLibFunc(&whisperModelNTextHead, libWhisper, cWhisperModelNTextHead)
	purego.RegisterLibFunc(&whisperModelNTextLayer, libWhisper, cWhisperModelNTextLayer)
	purego.RegisterLibFunc(&whisperModelNMels, libWhisper, cWhisperModelNMels)
	purego.RegisterLibFunc(&whisperModelFtype, libWhisper, cWhisperModelFtype)
	purego.RegisterLibFunc(&whisperModelType, libWhisper, cWhisperModelType)

	purego.RegisterLibFunc(&whisperGetLogits, libWhisper, cWhisperGetLogits)
	purego.RegisterLibFunc(&whisperGetLogitsFromState, libWhisper, cWhisperGetLogitsFromState)

	purego.RegisterLibFunc(&whisperTokenToStr, libWhisper, cWhisperTokenToStr)
	purego.RegisterLibFunc(&whisperModelTypeReadable, libWhisper, cWhisperModelTypeReadable)

	purego.RegisterLibFunc(&whisperTokenEot, libWhisper, cWhisperTokenEot)
	purego.RegisterLibFunc(&whisperTokenSot, libWhisper, cWhisperTokenSot)
	purego.RegisterLibFunc(&whisperTokenSolm, libWhisper, cWhisperTokenSolm)
	purego.RegisterLibFunc(&whisperTokenPrev, libWhisper, cWhisperTokenPrev)
	purego.RegisterLibFunc(&whisperTokenNosp, libWhisper, cWhisperTokenNosp)
	purego.RegisterLibFunc(&whisperTokenNot, libWhisper, cWhisperTokenNot)
	purego.RegisterLibFunc(&whisperTokenBeg, libWhisper, cWhisperTokenBeg)
	purego.RegisterLibFunc(&whisperTokenLang, libWhisper, cWhisperTokenLang)

	purego.RegisterLibFunc(&whisperTokenTranslate, libWhisper, cWhisperTokenTranslate)
	purego.RegisterLibFunc(&whisperTokenTranscribe, libWhisper, cWhisperTokenTranscribe)

	purego.RegisterLibFunc(&whisperPrintTimings, libWhisper, cWhisperPrintTimings)
	purego.RegisterLibFunc(&whisperResetTimings, libWhisper, cWhisperResetTimings)

	purego.RegisterLibFunc(&whisperPrintSystemInfo, libWhisper, cWhisperPrintSystemInfo)

	purego.RegisterLibFunc(&whisperContextDefaultParamsByRef, libWhisper, cWhisperContextDefaultParamsByRef)
	purego.RegisterLibFunc(&whisperContextDefaultParams, libWhisper, cWhisperContextDefaultParams)
	purego.RegisterLibFunc(&whisperFullDefaultParamsByRef, libWhisper, cWhisperFullDefaultParamsByRef)
	purego.RegisterLibFunc(&whisperFullDefaultParams, libWhisper, cWhisperFullDefaultParams)

	purego.RegisterLibFunc(&whisperFull, libWhisper, cWhisperFull)
	purego.RegisterLibFunc(&whisperFullWithState, libWhisper, cWhisperFullWithState)

	purego.RegisterLibFunc(&whisperFullParallel, libWhisper, cWhisperFullParallel)
	purego.RegisterLibFunc(&whisperFullNSegments, libWhisper, cWhisperFullNSegments)

	purego.RegisterLibFunc(&whisperFullNSegmentsFromState, libWhisper, cWhisperFullNSegmentsFromState)

	purego.RegisterLibFunc(&whisperFullLangId, libWhisper, cWhisperFullLangId)
	purego.RegisterLibFunc(&whisperFullLangIdFromState, libWhisper, cWhisperFullLangIdFromState)

	purego.RegisterLibFunc(&whisperFullGetSegmentT0, libWhisper, cWhisperFullGetSegmentT0)
	purego.RegisterLibFunc(&whisperFullGetSegmentT0FromState, libWhisper, cWhisperFullGetSegmentT0FromState)

	purego.RegisterLibFunc(&whisperFullGetSegmentT1, libWhisper, cWhisperFullGetSegmentT1)
	purego.RegisterLibFunc(&whisperFullGetSegmentT1FromState, libWhisper, cWhisperFullGetSegmentT1FromState)

	purego.RegisterLibFunc(&whisperFullGetSegmentSpeakerTurnNext, libWhisper, cWhisperFullGetSegmentSpeakerTurnNext)
	purego.RegisterLibFunc(&whisperFullGetSegmentSpeakerTurnNextFromState, libWhisper, cWhisperFullGetSegmentSpeakerTurnNextFromState)

	purego.RegisterLibFunc(&whisperFullGetSegmentText, libWhisper, cWhisperFullGetSegmentText)
	purego.RegisterLibFunc(&whisperFullGetSegmentTextFromState, libWhisper, cWhisperFullGetSegmentTextFromState)

	purego.RegisterLibFunc(&whisperFullNTokens, libWhisper, cWhisperFullNTokens)
	purego.RegisterLibFunc(&whisperFullNTokensFromState, libWhisper, cWhisperFullNTokensFromState)

	purego.RegisterLibFunc(&whisperFullGetTokenText, libWhisper, cWhisperFullGetTokenText)
	purego.RegisterLibFunc(&whisperFullGetTokenTextFromState, libWhisper, cWhisperFullGetTokenTextFromState)

	purego.RegisterLibFunc(&whisperFullGetTokenId, libWhisper, cWhisperFullGetTokenId)
	purego.RegisterLibFunc(&whisperFullGetTokenIdFromState, libWhisper, cWhisperFullGetTokenIdFromState)

	//purego.RegisterLibFunc(&whisperFullGetTokenData, libWhisper, cWhisperFullGetTokenData)
	//purego.RegisterLibFunc(&whisperFullGetTokenDataFromState, libWhisper, cWhisperFullGetTokenDataFromState)

	purego.RegisterLibFunc(&whisperFullGetTokenP, libWhisper, cWhisperFullGetTokenP)
	purego.RegisterLibFunc(&whisperFullGetTokenPFromState, libWhisper, cWhisperFullGetTokenPFromState)

	return &CWhisperImpl{
		libWhisper:                       libWhisper,
		cWhisperInitFromFileWithParams:   whisperInitFromFileWithParams,
		cWhisperInitFromBufferWithParams: whisperInitFromBufferWithParams,
		//whisperInitWithParams           func()

		cWhisperInitFromFileWithParamsNoState:   whisperInitFromFileWithParamsNoState,
		cWhisperInitFromBufferWithParamsNoState: whisperInitFromBufferWithParamsNoState,
		//whisperInitWithParamsNoState           func()

		cWhisperInitState:              whisperInitState,
		cWhisperCtxInitOpenvinoEncoder: whisperCtxInitOpenvinoEncoder,

		cWhisperFree:              whisperFree,
		cWhisperFreeState:         whisperFreeState,
		cWhisperFreeParams:        whisperFreeParams,
		cWhisperFreeContextParams: whisperFreeContextParams,

		cWhisperPcmToMel:          whisperPcmToMel,
		cWhisperPcmToMelWithState: whisperPcmToMelWithState,

		cWhisperPcmToMelPhaseVocoder:          whisperPcmToMelPhaseVocoder,
		cWhisperPcmToMelPhaseVocoderWithState: whisperPcmToMelPhaseVocoderWithState,

		cWhisperSetMel:          whisperSetMel,
		cWhisperSetMelWithState: whisperSetMelWithState,

		cWhisperEncode:          whisperEncode,
		cWhisperEncodeWithState: whisperEncodeWithState,

		cWhisperDecode:          whisperDecode,
		cWhisperDecodeWithState: whisperDecodeWithState,

		cWhisperTokenize: whisperTokenize,

		cWhisperLangMaxId: whisperLangMaxId,
		cWhisperLangStr:   whisperLangStr,

		cWhisperLangAutoDetect:          whisperLangAutoDetect,
		cWhisperLangAutoDetectWithState: whisperLangAutoDetectWithState,

		cWhisperNLen:           whisperNLen,
		cWhisperNLenFromState:  whisperNLenFromState,
		cWhisperNVocab:         whisperNVocab,
		cWhisperNTextCtx:       whisperNTextCtx,
		cWhisperNAudioCtx:      whisperNAudioCtx,
		cWhisperIsMultilingual: whisperIsMultilingual,

		cWhisperModelNVocab:      whisperModelNVocab,
		cWhisperModelNAudioCtx:   whisperModelNAudioCtx,
		cWhisperModelNAudioState: whisperModelNAudioState,
		cWhisperModelNAudioHead:  whisperModelNAudioHead,
		cWhisperModelNAudioLayer: whisperModelNAudioLayer,
		cWhisperModelNTextCtx:    whisperModelNTextCtx,
		cWhisperModelNTextState:  whisperModelNTextState,
		cWhisperModelNTextHead:   whisperModelNTextHead,
		cWhisperModelNTextLayer:  whisperModelNTextLayer,
		cWhisperModelNMels:       whisperModelNMels,
		cWhisperModelFtype:       whisperModelFtype,
		cWhisperModelType:        whisperModelType,

		cWhisperGetLogits:          whisperGetLogits,
		cWhisperGetLogitsFromState: whisperGetLogitsFromState,

		cWhisperTokenToStr:        whisperTokenToStr,
		cWhisperModelTypeReadable: whisperModelTypeReadable,

		cWhisperTokenEot:  whisperTokenEot,
		cWhisperTokenSot:  whisperTokenSot,
		cWhisperTokenSolm: whisperTokenSolm,
		cWhisperTokenPrev: whisperTokenPrev,
		cWhisperTokenNosp: whisperTokenNosp,
		cWhisperTokenNot:  whisperTokenNot,
		cWhisperTokenBeg:  whisperTokenBeg,
		cWhisperTokenLang: whisperTokenLang,

		cWhisperTokenTranslate:  whisperTokenTranslate,
		cWhisperTokenTranscribe: whisperTokenTranscribe,

		cWhisperPrintTimings: whisperPrintTimings,
		cWhisperResetTimings: whisperResetTimings,

		cWhisperPrintSystemInfo: whisperPrintSystemInfo,

		cWhisperContextDefaultParamsByRef: whisperContextDefaultParamsByRef,
		cWhisperContextDefaultParams:      whisperContextDefaultParams,
		cWhisperFullDefaultParamsByRef:    whisperFullDefaultParamsByRef,
		cWhisperFullDefaultParams:         whisperFullDefaultParams,

		cWhisperFull:          whisperFull,
		cWhisperFullWithState: whisperFullWithState,

		cWhisperFullParallel: whisperFullParallel,

		cWhisperFullNSegments:          whisperFullNSegments,
		cWhisperFullNSegmentsFromState: whisperFullNSegmentsFromState,

		cWhisperFullLangId: whisperFullLangId,

		cWhisperFullLangIdFromState: whisperFullLangIdFromState,

		cWhisperFullGetSegmentT0:          whisperFullGetSegmentT0,
		cWhisperFullGetSegmentT0FromState: whisperFullGetSegmentT0FromState,

		cWhisperFullGetSegmentT1:          whisperFullGetSegmentT1,
		cWhisperFullGetSegmentT1FromState: whisperFullGetSegmentT1FromState,

		cWhisperFullGetSegmentSpeakerTurnNext:          whisperFullGetSegmentSpeakerTurnNext,
		cWhisperFullGetSegmentSpeakerTurnNextFromState: whisperFullGetSegmentSpeakerTurnNextFromState,

		cWhisperFullGetSegmentText:          whisperFullGetSegmentText,
		cWhisperFullGetSegmentTextFromState: whisperFullGetSegmentTextFromState,

		cWhisperFullNTokens:          whisperFullNTokens,
		cWhisperFullNTokensFromState: whisperFullNTokensFromState,

		cWhisperFullGetTokenText:          whisperFullGetTokenText,
		cWhisperFullGetTokenTextFromState: whisperFullGetTokenTextFromState,

		cWhisperFullGetTokenId:          whisperFullGetTokenId,
		cWhisperFullGetTokenIdFromState: whisperFullGetTokenIdFromState,

		//TODO
		//whisperFullGetTokenData          func()
		//whisperFullGetTokenDataFromState func()

		cWhisperFullGetTokenP:          whisperFullGetTokenP,
		cWhisperFullGetTokenPFromState: whisperFullGetTokenPFromState,
	}, nil
}

func (c *CWhisperImpl) WhisperInitFromFileWithParams(pathModel string, params *WhisperContextParamsRef) *WhisperContext {
	ctx := c.cWhisperInitFromFileWithParams(pathModel, params.paramsRef)
	if ctx != 0 {
		return &WhisperContext{ctx: ctx}
	}
	return nil
}

func (c *CWhisperImpl) WhisperInitFromBufferWithParams(buffer []byte, params *WhisperContextParamsRef) *WhisperContext {
	ctx := c.cWhisperInitFromBufferWithParams(uintptr(unsafe.Pointer(&buffer[0])), len(buffer), params.paramsRef)
	return &WhisperContext{ctx: ctx}
}

func (c *CWhisperImpl) WhisperInitFromFileWithParamsNoState(pathModel string, params *WhisperContextParamsRef) *WhisperContext {
	ctx := c.cWhisperInitFromFileWithParamsNoState(pathModel, params.paramsRef)
	return &WhisperContext{ctx: ctx}
}

func (c *CWhisperImpl) WhisperInitFromBufferWithParamsNoState(buffer []byte, params *WhisperContextParamsRef) *WhisperContext {
	ctx := c.cWhisperInitFromBufferWithParamsNoState(uintptr(unsafe.Pointer(&buffer[0])), len(buffer), params.paramsRef)
	return &WhisperContext{ctx: ctx}
}

func (c *CWhisperImpl) WhisperInitState(ctx *WhisperContext) *WhisperState {
	state := c.cWhisperInitState(ctx.ctx)
	return &WhisperState{state: state}
}

func (c *CWhisperImpl) WhisperCtxInitOpenvinoEncoder(ctx *WhisperContext, modelPath, device, cacheDir string) *WhisperState {
	state := c.cWhisperCtxInitOpenvinoEncoder(ctx.ctx, modelPath, device, cacheDir)
	return &WhisperState{state: state}
}

func (c *CWhisperImpl) WhisperFree(ctx *WhisperContext) {
	c.cWhisperFree(ctx.ctx)
	if ctx.ctx != 0 {
		ctx.ctx = 0
	}
	ctx = nil
}

func (c *CWhisperImpl) WhisperFreeState(state *WhisperState) {
	c.cWhisperFreeState(state.state)
	if state.state != 0 {
		state.state = 0
	}
	state = nil
}

func (c *CWhisperImpl) WhisperFreeParams(params *WhisperFullParamsRef) {
	c.cWhisperFreeParams(params.paramsRef)
	if params.paramsRef != 0 {
		params.paramsRef = 0
	}
	params = nil
}

func (c *CWhisperImpl) WhisperFreeContextParams(params *WhisperContextParamsRef) {
	c.cWhisperFreeContextParams(params.paramsRef)
	if params.paramsRef != 0 {
		params.paramsRef = 0
	}
	params = nil
}

func (c *CWhisperImpl) WhisperPcmToMel(ctx *WhisperContext, samples []float32, nSamples, nThreads int) int {
	//TODO check this covert
	result := c.cWhisperPcmToMel(ctx.ctx, &samples[0], nSamples, nThreads)
	return result
}

func (c *CWhisperImpl) WhisperPcmToMelWithState(ctx *WhisperContext, state *WhisperState, samples []float32, nSamples, nThreads int) int {
	//TODO check this covert
	result := c.cWhisperPcmToMelWithState(ctx.ctx, state.state, &samples[0], nSamples, nThreads)
	return result
}

func (c *CWhisperImpl) WhisperPcmToMelPhaseVocoder(ctx *WhisperContext, samples []float32, nSamples, nThreads int) int {
	//TODO check this covert
	result := c.cWhisperPcmToMelPhaseVocoder(ctx.ctx, &samples[0], nSamples, nThreads)
	return result
}

func (c *CWhisperImpl) WhisperPcmToMelPhaseVocoderWithState(ctx *WhisperContext, state *WhisperState, samples []float32, nSamples, nThreads int) int {
	//TODO check this covert
	result := c.cWhisperPcmToMelPhaseVocoderWithState(ctx.ctx, state.state, &samples[0], nSamples, nThreads)
	return result
}

func (c *CWhisperImpl) WhisperSetMel(ctx *WhisperContext, data []float32, nLen, nMel int) int {
	//TODO check this covert
	result := c.cWhisperSetMel(ctx.ctx, &data[0], nLen, nMel)
	return result
}

func (c *CWhisperImpl) WhisperSetMelWithState(ctx *WhisperContext, state *WhisperState, data []float32, nLen, nMel int) int {
	//TODO check this covert
	result := c.cWhisperSetMelWithState(ctx.ctx, state.state, &data[0], nLen, nMel)
	return result
}

func (c *CWhisperImpl) WhisperEncode(ctx *WhisperContext, offset, nThreads int) int {
	//TODO check this covert
	result := c.cWhisperEncode(ctx.ctx, offset, nThreads)
	return result
}

func (c *CWhisperImpl) WhisperEncodeWithState(ctx *WhisperContext, state *WhisperState, offset, nThreads int) int {
	//TODO check this covert
	result := c.cWhisperEncodeWithState(ctx.ctx, state.state, offset, nThreads)
	return result
}

func (c *CWhisperImpl) WhisperDecode(ctx *WhisperContext, tokens []WhisperToken, nTokens, nPast, nThreads int) int {
	//TODO check this covert
	result := c.cWhisperDecode(ctx.ctx, (*int)(&tokens[0]), nTokens, nPast, nThreads)
	return result
}

func (c *CWhisperImpl) WhisperDecodeWithState(ctx *WhisperContext, state *WhisperState, tokens []WhisperToken, nTokens, nPast, nThreads int) int {
	//TODO check this covert
	result := c.cWhisperDecodeWithState(ctx.ctx, state.state, (*int)(&tokens[0]), nTokens, nPast, nThreads)
	return result
}

func (c *CWhisperImpl) WhisperTokenize(ctx *WhisperContext, text string, tokens []WhisperToken, nMaxTokens int) int {
	//TODO check this covert
	result := c.cWhisperTokenize(ctx.ctx, text, (*int)(&tokens[0]), nMaxTokens)
	return result
}

func (c *CWhisperImpl) WhisperLangMaxId() int {
	//TODO check this covert
	result := c.cWhisperLangMaxId()
	return result
}

func (c *CWhisperImpl) WhisperLangId(lang string) int {
	//TODO check this covert
	result := c.cWhisperLangId(lang)
	return result
}

func (c *CWhisperImpl) WhisperLangStr(id int) string {
	//TODO check this covert
	result := c.cWhisperLangStr(id)
	return result
}

func (c *CWhisperImpl) WhisperLangAutoDetect(ctx *WhisperContext, offsetMs, nThreads int, langProbs []float32) int {
	//TODO check this covert
	result := c.cWhisperLangAutoDetect(ctx.ctx, offsetMs, nThreads, &langProbs[0])
	return result
}

func (c *CWhisperImpl) WhisperLangAutoDetectWithState(ctx *WhisperContext, state *WhisperState, offsetMs, nThreads int, langProbs []float32) int {
	//TODO check this covert
	result := c.cWhisperLangAutoDetectWithState(ctx.ctx, state.state, offsetMs, nThreads, &langProbs[0])
	return result
}

func (c *CWhisperImpl) WhisperNLen(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperNLen(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperNLenFromState(state *WhisperState) int {
	//TODO check this covert
	result := c.cWhisperNLenFromState(state.state)
	return result
}

func (c *CWhisperImpl) WhisperNVocab(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperNVocab(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperNTextCtx(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperNTextCtx(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperNAudioCtx(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperNAudioCtx(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperIsMultilingual(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperNAudioCtx(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNVocab(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNVocab(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNAudioCtx(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNAudioCtx(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNAudioState(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNAudioState(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNAudioHead(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNAudioHead(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNAudioLayer(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNAudioLayer(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNTextCtx(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNTextCtx(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNTextState(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNTextState(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNTextHead(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNTextHead(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNTextLayer(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNTextLayer(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelNMels(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelNMels(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelFtype(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelFtype(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperModelType(ctx *WhisperContext) int {
	//TODO check this covert
	result := c.cWhisperModelType(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperGetLogits(ctx *WhisperContext) []float32 {
	//TODO finish this binding
	panic("WhisperFullGetTokenDataFromState is not support now")
	//c.cWhisperGetLogits(ctx.ctx)
	//return []float32{}
}

func (c *CWhisperImpl) WhisperGetLogitsFromState(state *WhisperState) []float32 {
	//TODO finish this binding
	panic("WhisperFullGetTokenDataFromState is not support now")
	//c.cWhisperGetLogitsFromState(state.state)
	//return []float32{}
}

func (c *CWhisperImpl) WhisperTokenToStr(ctx *WhisperContext, token WhisperToken) string {
	result := c.cWhisperTokenToStr(ctx.ctx, int(token))
	return result
}

func (c *CWhisperImpl) WhisperModelTypeReadable(ctx *WhisperContext) string {
	result := c.cWhisperModelTypeReadable(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperTokenEot(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenEot(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenSot(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenSot(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenSolm(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenSolm(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenPrev(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenPrev(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenNosp(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenNosp(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenNot(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenNot(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenBeg(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenBeg(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenLang(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenLang(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenTranslate(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenTranslate(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperTokenTranscribe(ctx *WhisperContext) WhisperToken {
	result := c.cWhisperTokenTranscribe(ctx.ctx)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperPrintTimings(ctx *WhisperContext) {
	c.cWhisperPrintTimings(ctx.ctx)
}

func (c *CWhisperImpl) WhisperResetTimings(ctx *WhisperContext) {
	c.cWhisperResetTimings(ctx.ctx)
}

func (c *CWhisperImpl) WhisperPrintSystemInfo() string {
	result := c.cWhisperPrintSystemInfo()
	return result
}

func (c *CWhisperImpl) WhisperContextDefaultParamsByRef() *WhisperContextParamsRef {
	paramsRef := c.cWhisperContextDefaultParamsByRef()
	return &WhisperContextParamsRef{
		paramsRef: paramsRef,
	}
}

func (c *CWhisperImpl) WhisperContextDefaultParams() *WhisperContextParams {
	params := c.cWhisperContextDefaultParams()
	return (*WhisperContextParams)(unsafe.Pointer(params))
}

func (c *CWhisperImpl) WhisperFullDefaultParamsByRef(strategy WhisperSamplingStrategy) *WhisperFullParamsRef {
	//i := int32(strategy)
	//paramsRef := c.cWhisperFullDefaultParamsByRef(uintptr(unsafe.Pointer(&i)))
	return &WhisperFullParamsRef{
		paramsRef: 0,
	}
}

func (c *CWhisperImpl) WhisperFullDefaultParams(strategy WhisperSamplingStrategy) *WhisperFullParams {
	//i := int32(strategy)
	//params := c.cWhisperFullDefaultParams(uintptr(unsafe.Pointer(&i)))
	return &WhisperFullParams{}
}

func (c *CWhisperImpl) WhisperFull(ctx *WhisperContext, params *WhisperFullParamsRef, samples []float32, nSamples int) int {
	result := c.cWhisperFull(ctx.ctx, params.paramsRef, &samples[0], nSamples)
	return result
}

func (c *CWhisperImpl) WhisperFullWithState(ctx *WhisperContext, state *WhisperState, params *WhisperFullParamsRef, samples []float32, nSamples int) int {
	result := c.cWhisperFullWithState(ctx.ctx, state.state, params.paramsRef, &samples[0], nSamples)
	return result
}

func (c *CWhisperImpl) WhisperFullParallel(ctx *WhisperContext, params *WhisperFullParamsRef, samples []float32, nSamples, nProcessors int) int {
	result := c.cWhisperFullParallel(ctx.ctx, params.paramsRef, &samples[0], nSamples, nProcessors)
	return result
}

func (c *CWhisperImpl) WhisperFullNSegments(ctx *WhisperContext) int {
	result := c.cWhisperFullNSegments(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperFullNSegmentsFromState(state *WhisperState) int {
	result := c.cWhisperFullNSegmentsFromState(state.state)
	return result
}

func (c *CWhisperImpl) WhisperFullLangId(ctx *WhisperContext) int {
	result := c.cWhisperFullNSegmentsFromState(ctx.ctx)
	return result
}

func (c *CWhisperImpl) WhisperFullLangIdFromState(state *WhisperState) int {
	result := c.cWhisperFullLangIdFromState(state.state)
	return result
}

func (c *CWhisperImpl) WhisperFullGetSegmentT0(ctx *WhisperContext, iSegment int) int64 {
	result := c.cWhisperFullGetSegmentT0(ctx.ctx, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullGetSegmentT0FromState(state *WhisperState, iSegment int) int64 {
	result := c.cWhisperFullGetSegmentT0FromState(state.state, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullGetSegmentT1(ctx *WhisperContext, iSegment int) int64 {
	result := c.cWhisperFullGetSegmentT1(ctx.ctx, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullGetSegmentT1FromState(state *WhisperState, iSegment int) int64 {
	result := c.cWhisperFullGetSegmentT1FromState(state.state, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullGetSegmentSpeakerTurnNext(ctx *WhisperContext, iSegment int) bool {
	result := c.cWhisperFullGetSegmentSpeakerTurnNext(ctx.ctx, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullGetSegmentSpeakerTurnNextFromState(state *WhisperState, iSegment int) bool {
	result := c.cWhisperFullGetSegmentSpeakerTurnNextFromState(state.state, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullGetSegmentText(ctx *WhisperContext, iSegment int) string {
	result := c.cWhisperFullGetSegmentText(ctx.ctx, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullGetSegmentTextFromState(state *WhisperState, iSegment int) string {
	result := c.cWhisperFullGetSegmentText(state.state, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullNTokens(ctx *WhisperContext, iSegment int) int {
	result := c.cWhisperFullNTokens(ctx.ctx, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullNTokensFromState(state *WhisperState, iSegment int) int {
	result := c.cWhisperFullNTokensFromState(state.state, iSegment)
	return result
}

func (c *CWhisperImpl) WhisperFullGetTokenText(ctx *WhisperContext, iSegment, iToken int) string {
	result := c.cWhisperFullGetTokenText(ctx.ctx, iSegment, iToken)
	return result
}

func (c *CWhisperImpl) WhisperFullGetTokenTextFromState(ctx *WhisperContext, state *WhisperState, iSegment, iToken int) string {
	result := c.cWhisperFullGetTokenTextFromState(ctx.ctx, state.state, iSegment, iToken)
	return result
}

func (c *CWhisperImpl) WhisperFullGetTokenId(ctx *WhisperContext, iSegment, iToken int) WhisperToken {
	result := c.cWhisperFullGetTokenId(ctx.ctx, iSegment, iToken)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperFullGetTokenIdFromState(state *WhisperState, iSegment, iToken int) WhisperToken {
	result := c.cWhisperFullGetTokenIdFromState(state.state, iSegment, iToken)
	return WhisperToken(result)
}

func (c *CWhisperImpl) WhisperFullGetTokenData(ctx *WhisperContext, iSegment, iToken int) WhisperTokenData {
	panic("WhisperFullGetTokenData is not support now")
}

func (c *CWhisperImpl) WhisperFullGetTokenDataFromState(state *WhisperState, iSegment, iToken int) WhisperTokenData {
	panic("WhisperFullGetTokenDataFromState is not support now")
}

func (c *CWhisperImpl) WhisperFullGetTokenP(ctx *WhisperContext, iSegment, iToken int) float32 {
	result := c.cWhisperFullGetTokenP(ctx.ctx, iSegment, iToken)
	return result
}

func (c *CWhisperImpl) WhisperFullGetTokenPFromState(state *WhisperState, iSegment, iToken int) float32 {
	result := c.cWhisperFullGetTokenPFromState(state.state, iSegment, iToken)
	return result
}
