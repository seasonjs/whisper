package whisper

import (
	"errors"
	"fmt"
	"github.com/go-audio/wav"
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
	fullParams *CWhisperFullParamsRef
}

func NewWhisperModel(dylibPath string, options WhisperOptions) (*WhisperModel, error) {
	cWhisper, err := NewCWhisper(dylibPath)
	if err != nil {
		return nil, err
	}
	fullParams := cWhisper.WhisperFullDefaultParamsByRef(WHISPER_SAMPLING_GREEDY)
	return &WhisperModel{
		cWhisper:   cWhisper,
		dylibPath:  dylibPath,
		options:    &options,
		fullParams: fullParams,
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

//func (m *WhisperModel) WhisperInitState() *WhisperState {
//	state := m.cWhisper.WhisperInitState(m.ctx)
//
//	return &WhisperState{
//		m:      m,
//		cState: state,
//	}
//}

func (m *WhisperModel) Predict(reader io.ReadSeeker) (string, error) {
	if m.ctx == nil {
		return "", errors.New("CWhisperContext is nil")
	}

	dec := wav.NewDecoder(reader)
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return "", err
	}
	if int(dec.SampleRate) != WHISPER_SAMPLE_RATE {
		return "", fmt.Errorf("unsupported sample rate: %d", dec.SampleRate)
	}
	if dec.NumChans != 1 {
		return "", fmt.Errorf("unsupported number of channels: %d", dec.NumChans)
	}

	data := buf.AsFloat32Buffer().Data

	if m.cWhisper.WhisperFullRefParallel(m.ctx, m.fullParams, data, len(data), m.options.NProcessors) != 0 {
		return "", errors.New("WhisperFullParallel Error,failed to process audio")
	}

	result := ""
	nSegments := m.cWhisper.WhisperFullNSegments(m.ctx)
	for i := 0; i < nSegments; i++ {
		text := m.cWhisper.WhisperFullGetSegmentText(m.ctx, i)
		result += text
	}

	return result, nil
}
