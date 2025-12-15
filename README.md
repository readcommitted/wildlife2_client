# 🐾 Wildlife Image Processing & Semantic Search System 🐾

The **Wildlife Image Processing & Semantic Search System** is a prototype platform for organizing, identifying, and exploring wildlife imagery using modern AI models, geospatial analysis, and semantic search.

This project combines **computer vision**, **location data**, and **embeddings** to help wildlife enthusiasts, researchers, and photographers document species, discover ecological patterns, and build richer digital field notes.

---

## ✨ Current Features

- ✅ **Species identification** using CLIP-based embeddings  
- ✅ **Regional species filtering** based on ecoregion boundaries  
- ✅ **Smart similarity search** with text prompts  
- ✅ **Location-aware ranking** to prioritize species likely present  
- ✅ **Image metadata extraction & geolocation** (via EXIF + OSM)  
- ✅ **Interactive Streamlit UI** for ingestion, validation, and analysis  
- ✅ **UMAP visualization tools** for exploring embedding space  
- ✅ **Early-stage SpeciesNet support** for wildlife classification  

---

## 🎯 Project Focus

This system is an **evolving foundation** designed to test core functionality and explore how AI can assist with **wildlife documentation and discovery**.  

The current version emphasizes **practical workflows** and **proof-of-concept features**, while the architecture is built for **future expansion**.

---

## 🛠 Technical Overview

- 🎨 **Streamlit UI** – Interactive workflows for ingestion & analysis  
- 🐘 **PostgreSQL + pgvector** – Storage and similarity search  
- 🧠 **OpenAI CLIP** – Image + text embeddings for semantic search  
- 🌍 **Species presence filtering** – Ecoregion-aware predictions  
- 🗺 **Location metadata enrichment** – OpenStreetMap + EXIF integration  
- 🔬 **LangChain & LangGraph** – Agentic orchestration & workflow testing  
- 🦾 **Custom models (SpeciesNet, YOLOv8)** – Early-stage supervised classifiers and smart cropping  

---

## 🚀 Future Enhancements

- 🌎 **Broader species detection models** and improved accuracy  
- 🏞 **Habitat-aware species ranking** with environmental context  
- 📍 **Expanded geospatial tools** (parks, sub-regions, polygons)  
- 🔎 **Enhanced semantic search** across image, text, and location fields  
- ✅ **Data quality validation & drift detection** for embeddings and models  
- 🔄 **Incremental retraining** with user feedback and new images  

---

## 📌 Notes

This project is under **active development**, with core workflows prioritized for testing and iteration.  
Contributions, ideas, and feedback are welcome as the system evolves into a more complete solution for **wildlife-focused image analysis**.  

---

## 🔌 Wildlife MCP API (Backend Service)
This project is supported by a companion backend service, the Wildlife MCP API, which powers geospatial queries, vector similarity search, ecoregion-aware species lookups, and agentic workflows.
The API provides:
 - Spatial queries using PostGIS
 - Embedding-based similarity search via pgvector
 - Species lookup by ecoregion and region codes
 - Tool endpoints designed for LangGraph and AI agent orchestration
🔌 API Repository: [https://github.com/readcommitted/wildlife_api](https://github.com/readcommitted/wildlife-api)

---

## 📖 References & Acknowledgments

This project builds on the work of several key tools, datasets, and communities:

- 🦉 **SpeciesNet** — Supervised wildlife classification  
- 🖼 **OpenAI CLIP** — Semantic image + text embeddings  
- 🐆 **Ultralytics YOLOv8** — Object detection & smart cropping  
- 🌍 **WWF WildFinder** — Ecoregion + species range database  
- 📸 **ExifTool by Phil Harvey** — Metadata extraction  

---

## 📜 License

MIT License – See [LICENSE](./LICENSE) for details.
