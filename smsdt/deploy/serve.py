import argparse
import numpy as np
import time
import os

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

try:
    import cv2
except ImportError:
    cv2 = None

# Logger for TensorRT
if trt:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTEngine:
    def __init__(self, engine_path):
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT or PyCUDA not installed. Cannot load engine.")

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Allocate buffers
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, input_data):
        # Copy input data to pagelocked memory
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # Transfer input data to the GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        return self.outputs[0]['host']


def main(args):
    if cv2 is None:
        print("OpenCV is not installed. Cannot process video.")
        return

    print(f"Loading TensorRT Engine from {args.engine_path}...")
    try:
        engine = TensorRTEngine(args.engine_path)
    except Exception as e:
        print(f"Failed to load engine: {e}")
        print("Running in mock mode for demonstration purposes...")
        engine = None

    print(f"Opening video source: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Error opening video stream or file.")
        return

    frames_buffer = []
    T = 8 # Number of frames per chunk

    # Pre-allocate dummy conf array for the pipeline
    dummy_conf = np.ones((1, T, 1), dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate Face Detection and Alignment (would use SCRFD here)
        # For demo, just resize the center crop
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        crop = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        resized = cv2.resize(crop, (224, 224))

        # Convert to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0)
        # HWC to CHW
        chw = np.transpose(normalized, (2, 0, 1))

        frames_buffer.append(chw)

        if len(frames_buffer) == T:
            # Prepare batch: Shape (1, T, C, H, W)
            batch = np.stack(frames_buffer, axis=0)
            batch = np.expand_dims(batch, axis=0)

            # The TRT engine expects inputs: x (1, T, C, H, W), conf (1, T, 1)
            # Actually, depending on ONNX export, it might just take x. Let's assume just x for now
            # if we exported with conf, we'd need to pack them or have multiple bindings.

            start_time = time.time()
            if engine is not None:
                output = engine.infer(batch)
                score = output[0]
            else:
                # Mock inference
                time.sleep(0.01) # Simulate ~10ms inference
                score = np.random.rand()

            latency = (time.time() - start_time) * 1000

            label = "FAKE" if score > 0.5 else "REAL"
            color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)

            print(f"Prediction: {label} (Score: {score:.4f}) - Latency: {latency:.1f}ms")

            # Draw on original frame (just the last one in the buffer)
            cv2.putText(frame, f"{label} {score:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"{latency:.1f}ms", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Clear buffer (sliding window could be implemented by popping first element)
            frames_buffer.clear()

        if args.show:
            cv2.imshow("S-MSDT Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone S-MSDT TRT Inference")
    parser.add_argument("--engine-path", type=str, default="outputs/smsdt_fp16.engine", help="Path to TensorRT engine")
    parser.add_argument("--source", type=str, default="0", help="Video source (RTSP url, video path, or camera id 0)")
    parser.add_argument("--show", action="store_true", help="Display video output")
    args = parser.parse_args()

    # If source is a digit, convert to int for webcam
    if args.source.isdigit():
        args.source = int(args.source)

    main(args)
