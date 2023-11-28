package whisper

import (
	"errors"
	"io"
)

type WhisperOptions struct {
	CpuThreads   int
	GpuEnable    bool
	NProcessors  int
	NoTimestamps bool
}

type WhisperModel struct {
	cWhisper   CWhisper
	dylibPath  string
	ctx        *CWhisperContext
	options    *WhisperOptions
	isAutoLoad bool
	params     *CWhisperFullParamsRef
}

func NewWhisperModel(dylibPath string, options WhisperOptions) (*WhisperModel, error) {
	cWhisper, err := NewCWhisper(dylibPath)
	if err != nil {
		return nil, err
	}
	params := cWhisper.WhisperFullDefaultParamsByRef(WHISPER_SAMPLING_GREEDY)
	//make sure NProcessors bigger than
	if options.NProcessors < 1 {
		options.NProcessors = 1
	}
	return &WhisperModel{
		cWhisper:  cWhisper,
		dylibPath: dylibPath,
		options:   &options,
		params:    params,
	}, nil
}

func (m *WhisperModel) LoadFromFile(path string) error {
	params := m.cWhisper.WhisperContextDefaultParamsByRef()
	ctx := m.cWhisper.WhisperInitFromFileWithParamsRef(path, params)
	if ctx == nil {
		return errors.New("LoadFormFile Error, can't get CWhisperContext")
	}
	m.ctx = ctx
	return nil
}

type WhisperState struct {
	m      *WhisperModel
	cState *CWhisperState
}

func (m *WhisperModel) Predict(reader io.ReadSeeker) (string, error) {
	if m.ctx == nil {
		return "", errors.New("CWhisperContext is nil")
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		return "", err
	}

	if m.cWhisper.WhisperFullRefParallel(m.ctx, m.params, data, m.options.NProcessors) != 0 {
		return "", errors.New("WhisperFullParallel Error, failed to process audio")
	}

	result := ""
	nSegments := m.cWhisper.WhisperFullNSegments(m.ctx)
	for i := 0; i < nSegments; i++ {
		text := m.cWhisper.WhisperFullGetSegmentText(m.ctx, i)
		result += text
	}

	return result, nil
}
