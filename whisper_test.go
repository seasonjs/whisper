package whisper_test

import (
	"os"
	"testing"
	"whisper"
)

func TestWhisper(t *testing.T) {
	model, err := whisper.NewWhisperModel(getLibrary(), whisper.WhisperOptions{
		CpuThreads: 4,
		GpuEnable:  false,
	})
	if err != nil {
		t.Error(err)
		return
	}
	err = model.LoadFromFile("./models/ggml-tiny.bin")
	if err != nil {
		t.Error(err)
		return
	}
	inFile, err := os.Open("./models/samples/jfk.wav")
	if err != nil {
		t.Error(err)
		return
	}
	defer inFile.Close()
	result, err := model.Predict(inFile)
	if err != nil {
		t.Error(err)
	}
	t.Log(result)
}
