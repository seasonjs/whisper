package whisper_test

import (
	"fmt"
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
		return "./deps/windows/whisper-abi.dll"
	default:
		panic(fmt.Errorf("GOOS=%s is not supported", runtime.GOOS))
	}
}
func TestNewCWhisper(t *testing.T) {
	model, err := whisper.NewCWhisper(getLibrary())
	if err != nil {
		t.Error(err)
		return
	}
	t.Run("Test WhisperContextDefaultParamsByRef", func(t *testing.T) {
		pointer := model.WhisperContextDefaultParamsByRef()
		t.Log(pointer)
	})

	t.Run("WhisperInitFromFileWithParamsRef", func(t *testing.T) {
		params := model.WhisperContextDefaultParamsByRef()
		t.Log(params)
		ctx := model.WhisperInitFromFileWithParamsRef("./models/ggml-tiny.bin", params)
		t.Log(ctx)
	})

	t.Run("Test WhisperFullDefaultParamsByRef", func(t *testing.T) {
		fullParams := model.WhisperFullDefaultParamsByRef(whisper.WHISPER_SAMPLING_GREEDY)
		t.Log(fullParams)
	})

}

func TestNativeSysCall(t *testing.T) {
	//// TODO why goland debug build pass but test build panic?
	//handle, err := windows.LoadLibrary(getLibrary())
	//if err != nil {
	//	t.Error(err)
	//	return
	//}
	//address, err := windows.GetProcAddress(handle,)
	//if err != nil {
	//	return
	//}
	//var i = int(WHISPER_SAMPLING_GREEDY)
	//r1, r2, _ := syscall.SyscallN(address, uintptr(i))
	////if err != nil {
	////	t.Error(err)
	////}
	//t.Log(r1, r2)
}
