import streamlit as st
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd # Import pandas for dataframe display


MODEL_NAME = "gpt2"
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
CACHE_SIMILARITY_THRESHOLD = 0.90
MAX_NEW_TOKENS = 50

if 'kv_cache' not in st.session_state:
    st.session_state.kv_cache = {}
    st.info("KV Cache initialized for this session.")

if 'history' not in st.session_state:
    st.session_state.history = []
    st.info("Generation history initialized for this session.")



@st.cache_resource
def load_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        return tokenizer, model, sentence_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

tokenizer, model, sentence_model = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sentence_model.to(device)



def get_embedding(text):
    sentence_model.to(device)
    return sentence_model.encode(text, convert_to_tensor=True, device=device)

def find_similar_cache(prompt_embedding, threshold, cache_dict):
    best_match_prompt = None
    best_match_cache = None
    max_similarity = -1.0

    if not cache_dict:
        return None, None, 0.0

    prompt_embedding_cpu_tuple = tuple(prompt_embedding.cpu().numpy())

    valid_keys = [k for k in cache_dict.keys() if isinstance(k, tuple) and len(k) > 0]
    if not valid_keys:
         return None, None, 0.0 # No valid keys

    try:
        cached_embeddings_list = [torch.tensor(k, device=device) for k in valid_keys]

        if not cached_embeddings_list:
             return None, None, 0.0

        cached_embeddings_tensor = torch.stack(cached_embeddings_list)

        similarities = util.pytorch_cos_sim(prompt_embedding.to(device).unsqueeze(0), cached_embeddings_tensor)[0]

        if similarities.numel() == 0:
             return None, None, 0.0

        best_match_idx = torch.argmax(similarities).item()
        max_similarity = similarities[best_match_idx].item()

        if max_similarity >= threshold:
            original_key = valid_keys[best_match_idx]
            if original_key in cache_dict:
                best_match_prompt, best_match_cache = cache_dict[original_key]
                return best_match_prompt, best_match_cache, max_similarity
            else:
                return None, None, max_similarity

    except Exception as e:
         return None, None, 0.0

    return None, None, max_similarity


def generate_text(prompt, use_cache=True, past_key_values=None):
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if input_ids.nelement() == 0 or input_ids.shape[-1] == 0:
        return "[Invalid Input: Tokenization Empty]", 0.0, None


    start_time = time.perf_counter()

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True
            )

        end_time = time.perf_counter()
        latency = end_time - start_time

        prompt_tokens_len = input_ids.shape[-1]
        if hasattr(outputs, 'sequences') and outputs.sequences.shape[0] > 0 and outputs.sequences.shape[1] >= prompt_tokens_len:
            generated_ids = outputs.sequences[0, prompt_tokens_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip() if hasattr(outputs, 'sequences') and outputs.sequences.shape[0] > 0 else "[Generation Error]"


        new_past_key_values = outputs.past_key_values if use_cache and hasattr(outputs, 'past_key_values') else None

        return generated_text, latency, new_past_key_values

    except IndexError as ie:
         # TODO: FIX THIS
         return "[Generation IndexError]", time.perf_counter() - start_time, None




st.title("Global KV Cache)")
st.markdown("""
KV Cache + Semantic Reuse for text generation using GPT-2 and Sentence Transformers.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Input & Controls")
    user_prompt = st.text_area("Enter your prompt here:", height=100, value="", key="prompt_input")
    similarity_threshold = st.slider("Similarity Threshold for Cache Reuse:", min_value=0.5, max_value=1.0, value=CACHE_SIMILARITY_THRESHOLD, step=0.01, key="threshold_slider")
    generate_button = st.button("Generate Text", key="generate_btn")
    clear_cache_button = st.button("Clear Cache & History", key="clear_btn")

    if clear_cache_button:
        st.session_state.kv_cache = {}
        st.session_state.history = []
        st.success("Cache and history cleared for this session!")
        st.rerun()


    st.header("KV Cache Store (Session)")
    st.markdown(f"Current number of cached items: **{len(st.session_state.kv_cache)}**")
    if st.session_state.kv_cache:
        cached_prompts = [details[0] for details in st.session_state.kv_cache.values()]
        with st.expander("View Cached Prompts"):
             st.json(cached_prompts)
    else:
        st.write("Cache is empty.")


with col2:
    st.header("Generation Results")
    if generate_button and user_prompt:
        current_embedding = get_embedding(user_prompt)
        current_embedding_tuple = tuple(current_embedding.cpu().numpy())

        st.write("---")
        st.write(f"**Processing Prompt:** '{user_prompt}'")
        with st.spinner("Searching cache..."):
            similar_prompt, cached_kv, similarity_score = find_similar_cache(
                current_embedding,
                similarity_threshold,
                st.session_state.kv_cache
            )
        cache_hit_success = False
        generation_details = {} # Initialize details dict

        if similar_prompt is not None:
            st.success(f"Cache Hit! Found similar prompt (Similarity: {similarity_score:.4f}): '{similar_prompt}'")
            st.info("Attempting to move cache to device and generate continuation...")

            cached_prompt_text, cached_kv_cpu = st.session_state.kv_cache[tuple(get_embedding(similar_prompt).cpu().numpy())]

            past_kv_on_device = None
            try:
                if cached_kv_cpu is not None:
                    past_kv_on_device = tuple(
                        tuple(pkv.to(device) for pkv in layer)
                        for layer in cached_kv_cpu
                    )

                with st.spinner("Generating with cache..."):
                    generated_text, latency, new_kv_cache = generate_text(
                        user_prompt,
                        use_cache=True,
                        past_key_values=past_kv_on_device 
                    )

                if new_kv_cache:
                    try:
                        past_kv_cpu = tuple(
                            tuple(pkv.detach().cpu() for pkv in layer)
                            for layer in new_kv_cache )
                        st.session_state.kv_cache[current_embedding_tuple] = (user_prompt, past_kv_cpu)
                    except Exception as e_store:
                         st.warning(f"Could not move/store new KV cache after hit: {e_store}")

                generation_details = {
                    "prompt": user_prompt, "status": "Cache Hit",
                    "similar_prompt_found": similar_prompt, "similarity": similarity_score,
                    "latency_ms": latency * 1000, "generated_text": generated_text,
                    "cache_size": len(st.session_state.kv_cache)
                }
                st.metric("Latency (with Cache)", f"{latency * 1000:.2f} ms")
                st.text_area("Generated Text (with Cache):", value=generated_text, height=150, key=f"gen_cache_{len(st.session_state.history)}")
                cache_hit_success = True

            except Exception as e_dev:
                st.warning(f"Cache Hit Failed: Could not move/use cached KV state for device {device}: {e_dev}. Falling back to cache miss.")

        if not cache_hit_success:
            if similar_prompt is None:
                 miss_reason = f"Cache Miss. No prompt found with similarity >= {similarity_threshold:.2f}. (Highest similarity found: {similarity_score:.4f})"
                 st.warning(miss_reason)
            else:
                 miss_reason = "Cache Miss: Failed processing reusable cache state."
                 st.warning(miss_reason)

            st.info("Generating from scratch...")
            with st.spinner("Generating without cache..."):
                 generated_text, latency, new_kv_cache = generate_text(
                     user_prompt,
                     use_cache=True,
                     past_key_values=None
                 )

            if new_kv_cache:
                try:
                    past_kv_cpu = tuple(
                        tuple(pkv.detach().cpu() for pkv in layer)
                        for layer in new_kv_cache )
                    st.session_state.kv_cache[current_embedding_tuple] = (user_prompt, past_kv_cpu)
                except Exception as e_store:
                     st.warning(f"Could not move/store new KV cache after miss: {e_store}")

            generation_details = {
                "prompt": user_prompt, "status": miss_reason.split('.')[0], # Short status
                "similar_prompt_found": None, "similarity": similarity_score,
                "latency_ms": latency * 1000, "generated_text": generated_text,
                "cache_size": len(st.session_state.kv_cache)
            }
            st.metric("Latency (No Cache Reuse)", f"{latency * 1000:.2f} ms")
            st.text_area("Generated Text (No Cache Reuse):", value=generated_text, height=150, key=f"gen_nocache_{len(st.session_state.history)}")

        if generation_details:
             st.session_state.history.append(generation_details)

    elif generate_button:
        st.warning("Please enter a prompt.")


    st.header("Generation History")
    if st.session_state.history:
         history_df = pd.DataFrame(st.session_state.history)
         display_df = history_df[[
             "prompt",
             "status",
             "latency_ms",
             "similarity",
             "similar_prompt_found",
             "cache_size", 
             "generated_text"
         ]].copy()

         display_df["latency_ms"] = display_df["latency_ms"].map('{:.2f}'.format)
         display_df["similarity"] = display_df["similarity"].map('{:.4f}'.format)
         display_df["similar_prompt_found"] = display_df["similar_prompt_found"].fillna("N/A")

         display_df.rename(columns={
            "latency_ms": "Latency (ms)",
            "status": "Cache Status",
            "similarity": "Max Similarity Found",
            "similar_prompt_found": "Matched Prompt (on Hit)",
            "cache_size": "Cache Size After Gen."
            }, inplace=True)
         st.dataframe(display_df.iloc[::-1], use_container_width=True)
    else:
        st.write("No generations yet in this session.")