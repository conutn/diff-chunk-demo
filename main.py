from pyscript import web, when

@when("click", "#submit-btn")
def translate_english(event):
    output_div = web.page["results"]
    output_div.innerText = "Pressed"