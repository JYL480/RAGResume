<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JYL480/RAGResume">
  </a>

<h3 align="center">Retrieval-Augmented Generation(RAG) Resume</h3>
<h3 align="center">Using all-mpnet-base-v2 emebdding model and Gemini LLM</h3>

  <p align="center">
    Retrieved information about myself!
    <br />
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

![image](https://github.com/user-attachments/assets/d8a66f9d-5036-4b99-b137-3b69c172da70)


Input any questions relating to me! The application will give you factual data about myself!

### Try it yourself

**Visit the link**: https://ragresume-ikyu8sh2pyjwx9wvjnkswh.streamlit.app/


### Features

- **Gemini-1.5-flash API**: Utilizes Google gemini-1.5-flash LLM API to handle input/query requests from users. 
- **User-friendly Interface**: Simple input interface for providing questions and associated context, with immediate response generation.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

Streamlit

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started
**Requirements**
- torch == 2.3.1
- google-generativeai == 0.7.2
- numpy == 1.26.4
- pandas == 2.2.2
- sentence_transformers == 3.0.1
- python-dotenv == 1.0.1

## Installation

Clone the repo:
```sh
git clone https://github.com/JYL480/RAGResume.git
```
Install requirements:
```sh
pip install -r requirements.txt
```
## Usage
```sh
streamlit run app.py
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
