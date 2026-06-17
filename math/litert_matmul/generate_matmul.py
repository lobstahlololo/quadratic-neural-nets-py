```python
import tensorflow as tf
import os

'''
Hacky method to get matmuls on gpu or dsp w.o writing custom code:
Create a .tflite, import it in LiteRT and run specifying gpu/dsp
'''
def create_matmul_int8(m, k, n, output_dir="."):
    input_a = tf.keras.layers.Input(shape=(m, k), batch_size=None, dtype=tf.int8, name='a')
    input_b = tf.keras.layers.Input(shape=(k, n), batch_size=None, dtype=tf.int8, name='b')
    a_fp = tf.cast(input_a, tf.float32)
    b_fp = tf.cast(input_b, tf.float32)
    output_fp = tf.matmul(a_fp, b_fp)
    output = tf.cast(output_fp, tf.int8)
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()
    filename = f"matmul_int8_{m}x{k}_{k}x{n}.tflite"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {filepath}")
    return filepath


def create_matmul_int16(m, k, n, output_dir="."):
    input_a = tf.keras.layers.Input(shape=(m, k), batch_size=None, dtype=tf.int16, name='a')
    input_b = tf.keras.layers.Input(shape=(k, n), batch_size=None, dtype=tf.int16, name='b')
    a_fp = tf.cast(input_a, tf.float32)
    b_fp = tf.cast(input_b, tf.float32)
    output_fp = tf.matmul(a_fp, b_fp)
    output = tf.cast(output_fp, tf.int16)
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    tflite_model = converter.convert()
    filename = f"matmul_int16_{m}x{k}_{k}x{n}.tflite"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {filepath}")
    return filepath


def generate_a16w8(m, k, n, output_dir="."):
    input_a = tf.keras.layers.Input(shape=(m, k), batch_size=None, dtype=tf.int16, name='a')
    input_b = tf.keras.layers.Input(shape=(k, n), batch_size=None, dtype=tf.int8, name='b')
    a_fp = tf.cast(input_a, tf.float32)
    b_fp = tf.cast(input_b, tf.float32)
    output_fp = tf.matmul(a_fp, b_fp)
    output = tf.cast(output_fp, tf.int16)
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    tflite_model = converter.convert()
    filename = f"matmul_a16w8_{m}x{k}_{k}x{n}.tflite"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {filepath}")
    return filepath


if __name__ == "__main__":
    create_matmul_int8(m=3, k=4, n=5)
    create_matmul_int16(m=3, k=4, n=5)
    generate_a16w8(m=3, k=4, n=5)
```