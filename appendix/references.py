"""
references.py — System References & Citations
----------------------------------------------

This Streamlit module provides proper attribution for third-party tools,
datasets, and research used in the Wildlife Vision System.

Includes:
- Citation for SpeciesNet (wildlife species identification model)
- Citation for ExifTool (image metadata extraction)
- Citation for WWF WildFinder (species occurrence data)

Maintaining accurate references ensures scientific credibility,
respect for open-source projects, and reproducibility for others
interested in the system's technical foundations.

Dependencies:
- Streamlit for display

"""

import streamlit as st

# --- SpeciesNet Citation ---
st.write("### SpeciesNet")
st.write("For species identification, SpeciesNet:")
st.code(
"""@article{gadot2024crop,
  title={To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images},
  author={Gadot, Tomer and Istrate, Ștefan and Kim, Hyungwon and Morris, Dan and Beery, Sara and Birch, Tanya and Ahumada, Jorge},
  journal={IET Computer Vision},
  year={2024},
  publisher={Wiley Online Library}
}"""
)

# --- ExifTool Citation ---
st.write("### ExifTool")
st.write("For image metadata processing, ExifTool:")
st.code("- ExifTool by Phil Harvey: https://exiftool.org")

# --- WWF WildFinder Citation ---
st.write("### World Wildlife Fund")
st.write("World Wildlife Fund. (n.d.). WildFinder database. World Wildlife Fund.")
st.write("https://www.worldwildlife.org/publications/wildfinder-database")

# --- OpenCLIP Citation ---
st.write("### OpenCLIP")
st.write("For embeddings, OpenCLIP:")
st.code(
"""@article{ilharco2021openclip,
  title={OpenCLIP: An open source implementation of CLIP},
  author={Ilharco, Gabriel and Wortsman, Mitchell and Gadre, Samir and others},
  journal={https://github.com/mlfoundations/open_clip},
  year={2021}
}"""
)

# --- Ultralytics YOLOv8 Citation ---
st.write("### YOLOv8")
st.write("For object detection and smart cropping, YOLOv8:")
st.code(
"""@software{yolov8_ultralytics,
  title={Ultralytics YOLOv8},
  author={Jocher, Glenn and others},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}"""
)

# --- pgvector Citation ---
st.write("### pgvector")
st.write("For vector similarity search, pgvector:")
st.code(
"""@software{pgvector,
  title={pgvector: Open-source vector similarity search for Postgres},
  author={pgvector contributors},
  year={2023},
  url={https://github.com/pgvector/pgvector}
}"""
)

# --- UMAP Citation ---
st.write("### UMAP")
st.write("For dimensionality reduction and embedding visualization, UMAP:")
st.code(
"""@article{mcinnes2018umap,
  title={UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction},
  author={McInnes, Leland and Healy, John and Melville, James},
  journal={arXiv preprint arXiv:1802.03426},
  year={2018}
}"""
)

# --- LangGraph Citation ---
st.write("### LangGraph")
st.write("For workflow orchestration and reasoning over embeddings, LangGraph:")
st.code(
"""@software{langgraph,
  title={LangGraph: Composable agent workflows for LLMs},
  author={LangChain Inc.},
  year={2024},
  url={https://www.langchain.com/langgraph}
}"""
)

# --- Streamlit Citation ---
st.write("### Streamlit")
st.write("For UI rendering, Streamlit:")
st.code(
"""@software{streamlit,
  title={Streamlit: The fastest way to build data apps},
  author={Streamlit Inc.},
  year={2022},
  url={https://streamlit.io}
}"""
)
