# CDF RAG Chatbot (V1)

A foundational Retrieval-Augmented Generation (RAG) chatbot built using official Community Dreams Foundation documentation as its sole knowledge source. The system retrieves relevant content from internal documents and generates answers that are explicitly grounded in those sources.

This repository focuses on correctness, traceability, and clarity. It is intended as a technical baseline rather than a production system.

---

## What This Project Does

- Ingests PDF and DOCX documentation
- Converts documents into structured, searchable chunks
- Builds a persistent vector index for semantic retrieval
- Retrieves relevant content for a user query
- Generates responses constrained to retrieved context
- Returns source references for transparency
- Exposes functionality via an API and a simple demo interface

---

## Why This Exists

This project exists to validate and demonstrate a complete RAG pipeline in isolation.  
It allows the core retrieval and generation logic to be evaluated independently before integration into the broader CDF portal.

---

## Project Scope (V1)

**Included**
- End-to-end RAG pipeline
- API for question answering
- Demo interface for validation
- Source-grounded responses
