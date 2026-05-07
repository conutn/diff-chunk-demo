from pyscript import web, when, fetch
from pyscript.ffi import to_js
import asyncio
import js

@when("click", "#submit-btn")
def button_submit(event):
    input_val = web.page["token"].value
    output_div = web.page["results"]
    output_div.innerText = input_val

def change_desc(event):
    select_val = web.page["modelselect"].value
    desc_div = web.page["modeldesc"]
    if (select_val == "standard256"): desc_div.innerText = "The standard transformer with a context window of 256 tokens. Should run faster than the chunked version when similar inputs are used."
    if (select_val == "chunked256"): desc_div.innerText = "The custom transformer with a context window of 256 tokens. Should run ap-proximately two times slower than the standard version."
    if (select_val == "standard1024"): desc_div.innerText = "Unimplemented."
    if (select_val == "chunked1024"): desc_div.innerText = "Unimplemented."

MODEL_URL = "https://drive.usercontent.google.com/download?id=1ylaP2uK77r-TBMv3WzKCS6aIUP18z_GD&export=download&confirm=t"
DATA_URL = "https://drive.usercontent.google.com/download?id=1tcP8rB-jKPVl6jH6G9cdbf-fX7VD1NUH&export=download&confirm=t"

async def fetch_bytes(url):
    try:
        response = await fetch(url)
        if not response.ok:
            print(f"Error: {response.status}")
        return await response.bytes()
    except Exception as e:
        print(f"Fetch error: {e}")
        return

async def load_model():
    print("Fetching .onnx...")
    model_bytes = await fetch_bytes(MODEL_URL)

    print("Fetching .onnx.data...")
    data_bytes = await fetch_bytes(DATA_URL)

    with open("/model.onnx", "wb") as f:
        f.write(model_bytes)
    with open("/model.onnx.data", "wb") as f:
        f.write(data_bytes)

    print("Creating ONNX session...")
    session = await js.ort.InferenceSession.create("/model.onnx")
    return session

async def run_inference(session, input_data):
    tensor = js.ort.Tensor.new("float32", to_js(input_data), to_js([1, len(input_data)]))
    feeds = js.Object.new()
    feeds[session.inputNames[0]] = tensor
    results = await session.run(feeds)
    output = results[session.outputNames[0]].data
    return output

asyncio.ensure_future(load_model())
