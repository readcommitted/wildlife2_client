"""
project.py — Wildlife Vision System Overview & Documentation
-------------------------------------------------------------

This Streamlit module presents an interactive project overview for the Wildlife Image Processing
and Semantic Search System. It provides structured, expandable sections covering:

- Project goals and motivation
- System architecture and workflow
- Data science and AI components
- Key features and benefits
- User guide for system usage
- Current limitations
- Future development roadmap

This overview is intended to onboard users, stakeholders, and collaborators, providing technical
and non-technical audiences with insight into system capabilities and design principles.

Dependencies:
- Streamlit for UI rendering

"""

import streamlit as st

# --- Project Overview Section ---
with st.expander("🔍 Project Overview"):
    st.write("""
   This project, **Wildlife Vision 2**, is an automated wildlife image processing and semantic search system.  
It streamlines the ingestion, classification, and retrieval of wildlife photographs — helping researchers, photographers, and enthusiasts manage large image collections with less manual effort.

  - **Why This Project?**
  - Manual tagging and organization of images is slow, inconsistent, and error-prone.  
  - **Automated species detection** (SpeciesNet + YOLO) reduces the need for manual identification.  
  - **Semantic search** (pgvector + OpenCLIP) enables finding images based on context and similarity rather than strict keywords.  
  - **Color and geospatial features** add richer dimensions for search and comparison.  
  - **User annotations and validations** provide a feedback loop that continuously improves the model and data quality.  
  - **Ad-Hoc Identification** lets users supply missing or corrected information when automation is uncertain.  
  - **Continuous learning**: these human-in-the-loop corrections feed back into retraining and future model improvements.  

    """)

# --- System Workflow Section ---
with st.expander("System Workflow"):
    st.write("""
1. **Image Ingestion:**  
   Images (JPEG/RAW) are uploaded from local devices or cloud storage.  
   EXIF metadata is extracted, and location is enriched with **OpenStreetMap geocoding**.  

2. **Object Detection & Cropping:**  
   **YOLOv8** detects animals in the frame and generates smart crops, improving downstream classification.  

3. **Species Detection:**  
   **SpeciesNet (CNN)** predicts species from cropped images, while embeddings (CLIP) provide similarity-based alternatives.  

4. **Embedding & Feature Extraction:**  
   Images are embedded into vector space (**pgvector + OpenCLIP**) for semantic search.  
   Additional features such as **color profiles** and **size class** are extracted for richer analysis.  

5. **Semantic Search & Retrieval:**  
   Users can search by species, colors, regions, or natural language queries.  
   Results are ranked by multimodal similarity (image + text + color + context).  

6. **User Interaction & Validation:**  
   Annotate, tag, or correct image species.  
   Human-in-the-loop validation feeds back into the knowledge base.  

7. **Ad-Hoc Identification:**  
   When automation is uncertain, users can manually tag species.  
   These identifications are logged and integrated into model improvement pipelines.  

8. **Continuous Learning:**  
   User annotations and corrections are aggregated for **future retraining**, improving accuracy over time.  

    """)

# --- Data Science, AI, and Engineering Section ---
with st.expander("Data Science, AI, and Engineering"):
    st.write("""
 - **Data Engineering:**  
  - Integration with **WWF and geospatial databases** for species distribution and habitat context.  
  - **Geocoding + polygons** (parks, valleys, regions) enrich metadata with precise spatial context.  
  - **Cloud ↔ Desktop Data Sync** keeps the canonical model in the cloud while enabling fast, offline-capable use on local desktop.  
  - **Dockerized desktop application** includes ingestion, EXIF/metadata extraction, and preprocessing modules.  
  - **APIs + LLM Access:** Canonical cloud model exposed through APIs for programmatic and LLM-based access.  

- **AI Engineering:**  
  - **SpeciesNet (CNN)** for supervised classification.  
  - **YOLOv8** for detection + smart cropping of animals in raw images.  
  - **OpenCLIP embeddings** for multimodal similarity (image ↔ text).  
  - **pgvector semantic search** in PostgreSQL for context-aware retrieval.  
  - **LangGraph + MCP (FastAPI-MCP)** orchestrates multi-model workflows, reranking, and reasoning steps.  
  - **Annotation-driven learning** integrates user feedback loops to improve embeddings and predictions.  

- **Model Training & Evaluation:**  
  - Transfer learning with **pretrained CNN backbones** to accelerate training on wildlife datasets.  
  - Balanced species sampling ensures robust generalization across classes.  
  - Hold-out test sets and **A/B splits** provide reliable accuracy evaluation.  
  - Continuous retraining pipelines incorporate user annotations, corrections, and new field data.  
  - Embedding validation and **UMAP/cluster analysis** used to monitor species separation quality.  

- **Software Engineering:**  
  - Modular Python architecture with **Streamlit** for UI.  
  - **PostgreSQL + pgvector** as the backbone for embeddings, metadata, and geospatial queries.  
  - Cloud-native storage with **DigitalOcean Spaces** for raw, JPEG, and processed image layers.  
  - Automated backups, WAL archiving, and database maintenance tools for reliability.  
  """)


# --- User Guide Section ---
with st.expander("User Guide"):
    st.write("""
    1. Upload images for processing.
    2. View processed images and detected species.
    3. Use search to find images by context (not just keywords).
    4. Annotate images with custom notes or tags.
    5. Use Ad-Hoc for identifying unknown species.
    """)

# --- Current Limitations Section ---
with st.expander("Current Limitations"):
    st.write("""
    - Species detection accuracy depends on the model (SpeciesNet).
    - Some images may lack EXIF metadata, affecting automatic tags.
    - Search may not be perfect for very obscure queries.
    - User-provided annotations can vary in quality.
    """)

# --- Future Plans Section ---
with st.expander("🚀 Future Plans"):
    st.write("""
- **Improve species detection** with more advanced and balanced models (e.g., larger CNNs, transformers, fine-tuned YOLO).  
- **Enhance semantic search performance and accuracy** by blending embeddings, geospatial, and color features.  
- **Add support for more species and image formats** (expanding coverage across mammals, birds, and beyond).  
- **Automate model retraining** using collected annotations and new ingestion data.  
- **Expand geospatial intelligence** with richer park/region polygons, seasonal ranges, and migration patterns.  
- **Integrate trust & explainability features** (confidence scoring, model comparison, and rationale).  
- **Multi-model fusion & orchestration**: leverage LangGraph + MCP to coordinate CNN, YOLO, embeddings, and LLM reasoning.  
- **Enable offline/edge workflows** with the Dockerized desktop app for field use where cloud access is limited.  
- **Broaden visualization tools** (e.g., UMAP/cluster maps, embedding space explorers, CLIP distance comparisons).  
- **Continuous data sync & backup** between local desktop and cloud canonical models with WAL archiving and snapshot management.  
- **Community & collaboration layer**: allow multiple users to share, validate, and contribute species data.  

    """)
