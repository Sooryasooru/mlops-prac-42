from app import train_model

def test_model():
    _,score = train_model()
    assert score > 0.5