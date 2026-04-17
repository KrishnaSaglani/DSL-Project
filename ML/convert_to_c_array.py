with open("model.tflite", "rb") as f:
    data = f.read()

with open("model_data.h", "w") as f:
    f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
    f.write("const unsigned char model_data[] = {\n")

    for i, byte in enumerate(data):
        if i % 12 == 0:
            f.write("\n ")
        f.write(f"0x{byte:02x}, ")

    f.write("\n};\n\n")
    f.write(f"const unsigned int model_data_len = {len(data)};\n\n")
    f.write("#endif\n")

print("model_data.h generated successfully!")