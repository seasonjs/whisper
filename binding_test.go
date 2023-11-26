package whisper

import (
	"fmt"
	"golang.org/x/sys/windows"
	"runtime"
	"syscall"
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
		t.Error(err)
		return
	}
	t.Run("Test C struct to Go struct", func(t *testing.T) {
		pointer := whisper.cWhisperContextDefaultParams()
		params := (*CWhisperContextParams)(unsafe.Pointer(pointer))
		t.Log(params)
	})

	t.Run("Test Go pointer to C struct", func(t *testing.T) {
		pointer := whisper.cWhisperContextDefaultParams()
		ctxPrt := whisper.cWhisperInitFromFileWithParams("./models/ggml-tiny.bin", pointer)
		t.Log(ctxPrt)
	})

	t.Run("Test Go enum to C enum", func(t *testing.T) {
		var i = int(WHISPER_SAMPLING_GREEDY)
		fullPrt := whisper.cWhisperFullDefaultParamsByRef(i)
		t.Log(fullPrt)
		params := (*CWhisperFullParams)(unsafe.Pointer(fullPrt))
		t.Log(params)
	})

	t.Run("Test C struct to Go uintptr then to Go struct", func(t *testing.T) {
		// TODO why goland debug build pass but test build panic?
		var i = int(WHISPER_SAMPLING_GREEDY)
		fullPrt := whisper.cWhisperFullDefaultParams(i)
		t.Log(fullPrt)
	})

}

func TestNativeSysCall(t *testing.T) {
	// TODO why goland debug build pass but test build panic?
	handle, err := windows.LoadLibrary(getLibrary())
	if err != nil {
		t.Error(err)
		return
	}
	address, err := windows.GetProcAddress(handle, cWhisperFullDefaultParams)
	if err != nil {
		return
	}
	var i = int(WHISPER_SAMPLING_GREEDY)
	r1, r2, _ := syscall.SyscallN(address, uintptr(i))
	//if err != nil {
	//	t.Error(err)
	//}
	t.Log(r1, r2)
}
