import gradio as gr
from gradio.themes import Ocean

with gr.Blocks(title='test', theme=Ocean(), css='.test { color: red; }') as demo:
    gr.Markdown('hello')

demo.launch(server_name='0.0.0.0', server_port=7860, share=False, show_error=True)
print('done')
