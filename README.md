<div align="center">
    <h1>Gordian.ai</h1>
    <p>An AI-powered smart `npm` package dependency visualizer and analyzer</p>
    <br />
    <img src="https://github.com/samueltckong/Gordian-AI/blob/main/demo.png">
    <br />
    <br />
    <a href="https://gordian-ai.streamlit.app">View Demo</a>
    &middot;
    <a href="https://github.com/shayna003/gordian.ai/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/shayna003/gordian.ai/issues/new?labels=enhancement">Request Feature</a>
</div>

## Features

## Getting Started
1. clone this repo and `cd` into it
```sh
git clone https://github.com/samueltckong/Gordian-AI/go.git
# or
git vlonr git@github.com:samueltckong/Gordian-AI.git 
```
```sh
cd Goridan-AI.ai
```
2. set up python environment
```sh
python3 -m venv gordian
source gordian/bin/activate
pip install -r requirements.txt
pip list
```
3. set up API keys
- rename `.env.example` file to `.env` and fill out the API keys:
```
GROQ_API_KEY="" 
BLACKBOX_API_KEY=""
```
- register an account on https://console.groq.com
- upgrade to developer plan on https://console.groq.com/settings/billing/plans
- go back to https://console.groq.com/landing/hackathon 
- click the "Apply for API credits" button and fill out a form
- when asked for the promo code, fill in `RAISE2025`
- wait for approval for API credits to arrive into your account
- you should then see 10$ API credits under "Coupons" of this page: https://console.groq.com/settings/billing/manage

4. run streamlit to host the website
```sh
streamlit run main.py
```

## Usage
For online package depencncy analysis, enter the name of a package on `npm`, such as `react`
For local project dependency analysis, upload a `package.json` or `package-lock.json` file. 
- Example file: [npm cli](https://github.com/npm/cli/blob/latest/package.json)

## License
Distributed under the MIT License.

## Built With
- Groq
- Llama
- Blackbox.ai
- Streamlit
- Python

## Our Team
We are from team [Lisa Mona](https://lablab.ai/event/raise-your-hack/lisa-mona-blackboxai-track) of [Lablab.ai Raise your Hack](https://lablab.ai/event/raise-your-hack)
- [Shayna003](https://github.com/shayna003)
- [samueltckong](https://github.com/samueltckong)
