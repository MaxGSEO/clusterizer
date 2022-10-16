import streamlit as st
from sentence_transformers import SentenceTransformer, util
import time
import chardet
import pandas as pd
from nltk import ngrams

# The code below is for the layout of the page
if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"

# region format
st.set_page_config(page_title="Semantic Keyword Clustering Tool - Patreon Beta 1", page_icon="✨", layout=layout)

beta_limit = 1000000
st.write("Web App Limited to First 10,000 Rows - Run Locally for More!")
st.title("Semantic Keyword Clustering Tool - Patreon Beta 1")

st.checkbox(
    "Widen layout",
    key="widen",
    help="Tick this box to change the layout to 'wide' mode",
)
st.caption("")

with st.form("my_form"):
    st.caption("")

    cols, col1, cole = st.columns([0.05, 1, 0.05])

    with col1:
        st.markdown("##### ① Upload your keyword report")

    cols, col1, cole = st.columns([0.2, 1, 0.2])

    with col1:
        uploaded_file = st.file_uploader(
            "CSV File in UTF-8 Format",
            help="""
    Which reports does the tool currently support?
    -   Input file must have a column named 'keyword'
    """,
        )

    cols, col1, cole = st.columns([0.2, 1, 0.2])

    st.markdown("")

    with st.expander("Advanced settings", expanded=True):
        st.markdown("")

        cols, col1, cole = st.columns([0.001, 1, 0.001])

        with col1:
            st.markdown("##### ② Pick Your Transformer Model")

        cols, col1, cole = st.columns([0.05, 1, 0.05])

        with col1:
            model_radio_button = st.radio(
                "Transformer model",
                [
                    "paraphrase-MiniLM-L3-v2",
                    "multi-qa-mpnet-base-dot-v1",
                    "paraphrase-multilingual-MiniLM-L12-v2",

                ],
                help="the model to use for the clustering",
            )

        cols, col1, cole = st.columns([0.2, 1, 0.2])

        with col1:
            st.markdown("***")

        cols, col1, cole = st.columns([0.001, 1, 0.001])

        with col1:
            st.write(
                "##### ③ Configure Your Cluster Settings",
                help="here you can configure the clustering settings",
            )

        st.caption("")

        # Three different columns:
        cols, col1, col2, cole = st.columns([0.1, 1, 1, 0.1])

        # You can also use "with" notation:
        with col1:
            accuracy_slide = st.slider(
                "Cluster accuracy: 0-100",
                value=85,
                help="the accuracy of the clusters",
            )

        with col2:
            min_cluster_size = st.slider(
                "Minimum Cluster size: 0-100",
                value=3,min_value=2,
                help="the minimum size of the clusters",
            )

        st.caption("")

        min_similarity = accuracy_slide / 100

        with col1:
            st.write("")
            remove_dupes = st.checkbox("Remove duplicate keywords?", value=True)

    st.write("")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

@st.cache(allow_output_mutation=True)
def get_model():
    model = SentenceTransformer(model_radio_button)

    return model

model = get_model()

if uploaded_file is not None:

    try:

        result = chardet.detect(uploaded_file.getvalue())
        encoding_value = result["encoding"]

        if encoding_value == "UTF-16":
            white_space = True
        else:
            white_space = False

        df = pd.read_csv(
            uploaded_file,
            encoding=encoding_value,
            delim_whitespace=white_space,
            on_bad_lines='skip',
        )

        number_of_rows = len(df)

        if number_of_rows > beta_limit:
            df = df[:beta_limit]
            st.caption(
                "🚨 Imported rows over the beta limit, limiting to first "
                + str(beta_limit)
                + " rows."
            )

        if number_of_rows == 0:
            st.caption("Your sheet seems empty!")
    except UnicodeDecodeError:
        st.warning(
            """
            🚨 The file doesn't seem to load. Check the filetype, file format and Schema

            """
        )

else:
    st.stop()

with st.form(key="columns_in_form_2"):
    # Three different columns:
    cols, col1, col2, cole = st.columns([0.05, 1, 1, 0.05])

    # You can also use "with" notation:
    with col1:

        st.subheader("")
        st.caption("")
        st.markdown("##### ④ Select the column to cluster 👉")

    with col2:
        kw_col = st.selectbox("", df.columns)
    st.write("")

    cols, col1, cole = st.columns([0.0025, 1, 0.03])

    with col1:
        with st.expander("View imported data (pre-clustering)", expanded=False):
            st.write(df)

    st.caption("")

    submitted = st.form_submit_button("Submit")
    df.rename(columns={kw_col: 'keyword', "spoke": "spoke Old"}, inplace=True)

# store the data
cluster_name_list = []
corpus_sentences_list = []
df_all = []

if submitted:
    if remove_dupes:
        df.drop_duplicates(subset='keyword', inplace=True)

    # start timing the script
    startTime = time.time()
    st.info("Clustering keywords. This may take a while!")

    corpus_set = set(df['keyword'])
    corpus_set_all = corpus_set

    cluster = True
    while cluster:
        try:

            corpus_sentences = list(corpus_set)
            check_len = len(corpus_sentences)

            corpus_embeddings = model.encode(
                corpus_sentences,
                batch_size=1024,
                show_progress_bar=True,
                convert_to_tensor=False,
            )
            clusters = util.community_detection(
                corpus_embeddings,
                min_community_size=min_cluster_size,
                threshold=min_similarity,
            )
        except RuntimeError:
            st.warning("Nothing to cluster! Did you select the right column?")
            st.stop()
        for keyword, cluster in enumerate(clusters):
            for sentence_id in cluster[0:]:
                corpus_sentences_list.append(corpus_sentences[sentence_id])
                cluster_name_list.append(
                    "Cluster {}, #{} Elements ".format(keyword + 1, len(cluster))
                )

        df_new = pd.DataFrame(None)
        df_new["spoke"] = cluster_name_list
        df_new['keyword'] = corpus_sentences_list
        df_all.append(df_new)
        have = set(df_new['keyword'])
        corpus_set = corpus_set_all - have
        remaining = len(corpus_set)

        if check_len == remaining:
            break

    print("Finished Clustering!")
    df_new = pd.concat(df_all)
    df = df.merge(df_new.drop_duplicates('keyword'), how="left", on='keyword')
    # -------------------- create unigrams for each cluster to be used as the parent_topic -----------------------------
    df_unigram = (
        df.assign(
            keyword=[[" ".join(x) for x in ngrams(s.split(), n=1)] for s in df["keyword"]]
        )
            .explode("keyword")
            .groupby("spoke")["keyword"]
            .apply(lambda g: g.mode())
            .reset_index(name="hub")
    )

    df = df.merge(df_unigram.drop_duplicates('spoke'), how='left', on="spoke")

    df_bigram = (
        df.assign(
            keyword=[[" ".join(x) for x in ngrams(s.split(), n=2)] for s in df["keyword"]]
        )
            .explode("keyword")
            .groupby("spoke")["keyword"]
            .apply(lambda g: g.mode())
            .reset_index(name='bigrams')
    )

    df = df.merge(df_bigram.drop_duplicates('spoke'), how='left', on="spoke")
    df['spoke'] = df['bigrams']

    # assign no_cluster values and delete helper columns
    df['hub'] = df['hub'].fillna("no_cluster")
    df['spoke'] = df['spoke'].fillna("no_cluster")

    del df['bigrams']
    del df['level_1_x']
    del df['level_1_y']
    
    # ------------------------------ pop the spoke column to the front ------------------------------------------

    col = df.pop('keyword')
    df.insert(0, col.name, col)
    col = df.pop('spoke')
    df.insert(0, col.name, col)
    col = df.pop('hub')
    df.insert(0, col.name, col)
    df.sort_values(["hub", "spoke", 'keyword'], ascending=[True, True, True], inplace=True)

    # -------------------------------------- download csv file ---------------------------------------------------------

    st.success(
        "All keywords clustered successfully. Took {0} seconds!".format(
            time.time() - startTime
        )
    )
    df_csv = df.copy()  # make unique csv for download (so that search volume data can be unmerged)

    try:
        df['keyword'] = df['keyword'].str.split("\(").str[0].str.strip()
        df['spoke'] = df['spoke'].str.split("\(").str[0].str.strip()
    except Exception:
        pass

    def convert_df(df):
        return df_csv.to_csv(index=False).encode("utf-8")

    # download results
    st.markdown("### **🎈 Download a CSV Export!**")
    st.write("")
    csv = convert_df(df)
    st.download_button(
        label="📥 Download your report!",
        data=csv,
        file_name="your_keywords_clustered.csv",
        mime="text/csv",
    )
