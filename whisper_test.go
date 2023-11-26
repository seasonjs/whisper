package whisper_test

import (
	"fmt"
	"os"
	"runtime"
	"testing"
	"whisper"
)

func getLibrary() string {
	switch runtime.GOOS {
	//case "darwin":
	//	return "./deps/darwin/librwkv_arm64.dylib"
	//case "linux":
	//	return "./deps/linux/librwkv.so"
	case "windows":
		return "./deps/windows/whisper.dll"
	default:
		panic(fmt.Errorf("GOOS=%s is not supported", runtime.GOOS))
	}
}

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
