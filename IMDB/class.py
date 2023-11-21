import tensorflow as tf
import tensorflow_datasets as tfds
import time

# 1. Install TensorFlow (if not already installed):
# !pip install tensorflow

# 2. Load the IMDb dataset
def load_data():
    (train_data, test_data), info = tfds.load(
        'imdb_reviews/plain_text',
        split = ('train', 'test'),
        as_supervised=True,
        with_info=True
    )
    return train_data, test_data

# 3. Preprocess the data
def preprocess_data(train_data, test_data):
    tokenizer = tfds.features.text.Tokenizer()
    
    # Building vocabulary
    vocabulary_set = set()
    for text, _ in train_data:
        some_tokens = tokenizer.tokenize(text.numpy())
        vocabulary_set.update(some_tokens)
    
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    # Encode the text
    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    train_data = train_data.map(encode_map_fn)
    test_data = test_data.map(encode_map_fn)

    return train_data, test_data

# 4. Build the classification model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 5. Train and evaluate the model
def train_evaluate_model(train_data, test_data, use_gpu=True):
    if use_gpu:
        device_name = '/GPU:0'
    else:
        device_name = '/CPU:0'

    with tf.device(device_name):
        model = build_model()
        start_time = time.time()
        model.fit(train_data, epochs=10)
        duration = time.time() - start_time
        loss, accuracy = model.evaluate(test_data)

    return duration, loss, accuracy

# Main execution
train_data, test_data = load_data()
train_data, test_data = preprocess_data(train_data, test_data)

# Train on CPU
cpu_duration, cpu_loss, cpu_accuracy = train_evaluate_model(train_data, test_data, use_gpu=False)
print(f"Training on CPU took {cpu_duration} seconds.")

# Train on GPU
gpu_duration, gpu_loss, gpu_accuracy = train_evaluate_model(train_data, test_data, use_gpu=True)
print(f"Training on GPU took {gpu_duration} seconds.")

# 6. Compare the speeds
print(f"Speedup: {cpu_duration / gpu_duration}x")
