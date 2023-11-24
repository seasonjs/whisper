package whisper

import "errors"

type WhisperOptions struct {
	CpuThreads int
	GpuEnable  bool
}

type WhisperModel struct {
	cWhisper   CWhisper
	dylibPath  string
	ctx        *WhisperContext
	options    *WhisperOptions
	isAutoLoad bool
}

func NewWhisperModel(dylibPath string, options WhisperOptions) (*WhisperModel, error) {
	cWhisper, err := NewCWhisper(dylibPath)
	if err != nil {
		return nil, err
	}

	return &WhisperModel{
		cWhisper:  cWhisper,
		dylibPath: dylibPath,
		options:   &options,
	}, nil
}

func (m *WhisperModel) LoadFromFile(path string) error {
	params := WhisperContextParams{UseGpu: m.options.GpuEnable}
	ctx := m.cWhisper.WhisperInitFromFileWithParams(path, params.ToWhisperContextParamsRef())
	if ctx == nil {
		return errors.New("LoadFormFile Error, can't get WhisperContext")
	}
	m.ctx = ctx
	return nil
}
