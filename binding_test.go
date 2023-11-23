package whisper

import (
	"fmt"
	"runtime"
	"testing"
	"unsafe"
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
func TestNewCWhisper(t *testing.T) {
	whisper, err := NewCWhisper(getLibrary())
	if err != nil {
		return
	}
	t.Run("Test C struct to Go struct", func(t *testing.T) {
		pointer := whisper.cWhisperContextDefaultParams()
		params := (*WhisperContextParams)(unsafe.Pointer(pointer))
		t.Log(params)
	})

	t.Run("Test Go pointer to C struct", func(t *testing.T) {
		pointer := whisper.cWhisperContextDefaultParams()
		ctxPrt := whisper.cWhisperInitFromFileWithParams("./models/ggml-tiny.bin", pointer)
		t.Log(ctxPrt)
	})

	t.Run("Test Go enum to C enum", func(t *testing.T) {
		var i = int32(WHISPER_SAMPLING_GREEDY)
		fullPrt := whisper.cWhisperFullDefaultParams(uintptr(unsafe.Pointer(&i)))
		t.Log(fullPrt)
		params := (*WhisperFullParams)(unsafe.Pointer(fullPrt))
		t.Log(params)
	})

}
