from streamlit_webrtc import webrtc_streamer
import av

def test_callback(frame):
    return frame

webrtc_streamer(
    key="test",
    media_stream_constraints={"video": True, "audio": False},
)
