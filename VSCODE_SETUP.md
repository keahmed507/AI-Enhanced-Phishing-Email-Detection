# Running the Phishing Detection Project in VS Code

This covers getting the backend (Flask API + trained models) and the
frontend (`index.html`) running locally, side by side, in VS Code.

---

## Prerequisites

- **Python 3.9–3.13** installed and on your PATH
  Check with:
  ```bash
  python --version
  ```
- **VS Code** with the **Python extension** installed (Extensions
  panel → search "Python" → the Microsoft one)
- The two delivered files: `phishing-detection-backend.zip` and
  `index.html`

---

## 1. Unzip the backend

Extract `phishing-detection-backend.zip` anywhere you like. You'll get
a `backend/` folder containing the code, real datasets, and
pre-trained models.

## 2. Open the folder in VS Code

**File → Open Folder…** → select the extracted `backend` folder.

## 3. Open the integrated terminal

**Terminal → New Terminal** (or `` Ctrl+` ``). It opens already
pointed at `backend` — no `cd` needed.

## 4. Create a virtual environment

```bash
python -m venv venv
```

## 5. Select the interpreter

VS Code usually pops up a notification: *"We noticed a new
environment has been created. Do you want to select it?"* → click
**Yes**.

If it doesn't appear:
`Ctrl+Shift+P` (or `Cmd+Shift+P`) → **"Python: Select Interpreter"** →
choose the one showing `.\venv\Scripts\python.exe` (Windows) or
`./venv/bin/python` (Mac/Linux).

## 6. Open a fresh terminal so the venv activates

Close the terminal and open a new one (**Terminal → New Terminal**).
You should see `(venv)` at the start of the prompt. If it's not
active, activate manually:

- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **Windows (cmd):** `venv\Scripts\activate.bat`
- **Mac/Linux:** `source venv/bin/activate`

> If PowerShell blocks the activation script, run this once first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

## 7. Install dependencies

```bash
pip install -r requirements.txt
```

Takes a minute or two — TensorFlow is the largest download.

> **Windows + Python 3.13, seeing a `meson`/`vswhere.exe` build
> error?** That means pip is trying to compile scikit-learn from
> source because an older version has no Python 3.13 wheel. The
> `requirements.txt` included with this project already uses
> versions with 3.13 wheels
> (`scikit-learn>=1.6.0`, `numpy>=2.1`, `pandas>=2.2.3,<3.0`), so a
> fresh `pip install -r requirements.txt` should not hit this. If it
> still does, delete `venv` and recreate it, then reinstall.

## 8. Run the backend

Either:
- In the terminal: `python app.py`, **or**
- Open `app.py` in the editor and press **F5** (uses the selected
  venv interpreter automatically)

You should see:
```
* Serving Flask app 'app'
* Debug mode: off
* Running on http://127.0.0.1:5000
```

**Leave this terminal/process running** for as long as you want the
backend available.

## 9. Open the frontend

`index.html` is separate from the `backend` folder. Just find it in
your file explorer and **double-click it** — it opens in your default
browser. You don't need VS Code for this step at all.

## 10. Confirm the connection

On the page, look at the small status light in the console panel
header (next to "triage_console.py"). It should read
**"backend connected"** within a second or two.

If it says **"local demo mode (backend offline)"** instead:
- Check the VS Code terminal from step 8 — is `app.py` still running
  without errors?
- Make sure nothing else changed `BACKEND_URL` in `index.html` away
  from `http://127.0.0.1:5000`

## 11. Use it

- Click one of the **"Load: ..."** sample buttons, or paste your own
  email text
- Pick a **sender profile** and a **model** (Logistic Regression, SVM,
  Random Forest, or LSTM) from the dropdowns
- Click **"Run classification"**

You'll get a verdict, a confidence score, content/behavior risk
meters, and the specific features (urgency words, URLs, etc.) that
were detected.

---

## Stopping it

In the VS Code terminal running `app.py`, press **Ctrl+C**. Closing
`index.html`'s browser tab doesn't affect the backend — stop it
separately if you're done.

## Starting it again later

You don't need to repeat the whole setup — just:
```bash
cd backend
venv\Scripts\activate    # or: source venv/bin/activate
python app.py
```
then open `index.html` again.
