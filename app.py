from flask import Flask, render_template, request, make_response, jsonify
from transformer import *

import torch

app = Flask(__name__, static_url_path = "/static")

text = "In deep learning, the transformer is a family of artificial neural network architectures based on the multi-head attention mechanism, in which text is converted to numerical representations called tokens, and each token is converted into a vector via lookup from a word embedding table. At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism, allowing the signal for key tokens to be amplified and less important tokens to be diminished. Transformers have the advantage of having no recurrent units, therefore requiring less training time than earlier recurrent neural architectures (RNNs) such as long short-term memory (LSTM). Later variations have been widely adopted for training large language models (LLMs) on large (language) datasets. The original version of the transformer architecture was proposed in the 2017 paper \"Attention Is All You Need\" by researchers at Google. The predecessors of transformers were developed as an improvement over previous architectures for machine translation, but have found many applications since. They are used in large-scale natural language processing, computer vision (vision transformers), reinforcement learning, audio, multimodal learning, robotics, and playing chess. It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs) and BERT (bidirectional encoder representations from transformers). For many years, sequence modelling and generation was done by using plain recurrent neural networks (RNNs). A well-cited early example was the Elman network (1990). In theory, the information from one token can propagate arbitrarily far down the sequence, but in practice the vanishing-gradient problem leaves the model's state at the end of a long sentence without precise, extractable information about preceding tokens. A key breakthrough was LSTM (1995), an RNN which used various innovations to overcome the vanishing gradient problem, allowing efficient learning of long-sequence modelling. One key innovation was the use of an attention mechanism which used neurons that multiply the outputs of other neurons, so-called multiplicative units. Neural networks using multiplicative units were later called sigma-pi networks or higher-order networks. LSTM became the standard architecture for long sequence modelling until the 2017 publication of transformers. However, LSTM still used sequential processing, like most other RNNs. Specifically, RNNs operate one token at a time from first to last; they cannot operate in parallel over all tokens in a sequence. Modern transformers overcome this problem, but unlike RNNs, they require computation time that is quadratic in the size of the context window. The linearly scaling fast weight controller (1992) learns to compute a weight matrix for further processing depending on the input. One of its two networks has \"fast weights\" or \"dynamic links\" (1981). A slow neural network learns by gradient descent to generate keys and values for computing the weight changes of the fast neural network which computes answers to queries. This was later shown to be equivalent to the unnormalized linear transformer. "

vocab = read_vocab("vocab.txt")
vocab_size = len(vocab)
vocab_dct = get_vocab_dct_encoder(vocab)

tokenizer = Tokenizer(vocab, "[UNK]")
template = tokenizer.tokenize(text, is_wiki = False)

hier_dct_basic = {
    "model_type": "HierarchicalTransformer",
    "emb_dim": 768,
    "hidden_dim": 3072,
    "nhead": 12,
    "token_layers": 1,
    "chunk_layers": 8,
    "vocab_size": 8018,
    "max_chunks": 8,
    "beta": 5,
    "max_len": 256,
    "dropout": 0
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model("train/chunk768.pth", hier_dct_basic, device)
model.eval()

@app.route("/")
def root():
    return render_template("index.html")

@app.route("/request_model", methods = ["POST"])
def model_request():
    data = request.get_json()
    words = data["words"]
    tokens = tokenizer.tokenize(words, is_wiki = False)
    lst = []

    vectors = []
    actual = []
    comb_tokens = template + tokens

    input_idxs = [list(map(vocab_dct.get, comb_tokens))[-model.max_len:]]
    ret = input_idxs[0][:]
    
    with torch.no_grad():
        for i in range(10):
            print(i)
            input_tensor = torch.tensor(input_idxs).to(device)
            mask = build_mask(input_tensor).to(device)
            pred = model(input_tensor, mask = mask)
            pred = torch.softmax(pred[0][-1], 0)
            sampled_idx = torch.multinomial(pred, num_samples = 1).detach().clone().item()
            ret.append(sampled_idx)
            input_idxs[0].append(sampled_idx)
            input_idxs[0] = input_idxs[0][-model.max_len:]

    ret = ret[-len(tokens)-10:]

    r = " ".join(list(map(vocab.__getitem__, ret)))
    r = r.replace(" §", "")
    return jsonify(r)    

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 80, debug = True)