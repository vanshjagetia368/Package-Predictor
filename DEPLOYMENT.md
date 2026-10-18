# 🚀 Deployment Guide: GitHub → Render

This takes the project from a folder on your computer to a public URL.

---

## Part 1: Put the project on GitHub

### 1. Install Git
Download from https://git-scm.com/downloads and install with the defaults. Check it worked:

```bash
git --version
```

### 2. Create a GitHub account
Sign up at https://github.com if you do not have one.

### 3. Create an empty repository
1. Click the **+** at the top right of GitHub → **New repository**
2. **Repository name:** `placement-predictor`
3. Set it to **Public**
4. Leave **Add a README**, **.gitignore** and **license** all **unchecked** (the project already has them)
5. Click **Create repository**

### 4. Push the project
Open a terminal **inside the project folder** (the one containing `app.py`) and run:

```bash
git init
git add .
git commit -m "Add CGPA to package predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/placement-predictor.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If Git asks you to sign in, use your GitHub login in the browser window, or a Personal Access Token as the password (GitHub → Settings → Developer settings → Personal access tokens).

### 5. Check it
Refresh the repository page. You should see `app.py`, `model.py`, `requirements.txt`, `render.yaml`, `data/`, `assets/` and the rest.

---

## Part 2: Host it on Render

### 1. Create a Render account
Go to https://render.com → **Get Started** → **Sign up with GitHub**. Authorising with GitHub makes the next steps easier.

### 2. Create a Web Service
1. In the Render dashboard click **New +** → **Web Service**
2. Choose **Build and deploy from a Git repository** → **Next**
3. If your repository is not listed, click **Configure account** and grant Render access to `placement-predictor`
4. Click **Connect** next to `placement-predictor`

### 3. Fill in the settings

| Field | Value |
| --- | --- |
| **Name** | `placement-package-predictor` (this becomes part of your URL) |
| **Region** | the one closest to you |
| **Branch** | `main` |
| **Root Directory** | leave empty |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| **Instance Type** | `Free` |

### 4. Set the Python version
Scroll to **Environment Variables** → **Add Environment Variable**:

| Key | Value |
| --- | --- |
| `PYTHON_VERSION` | `3.12.8` |

This matters: the pinned packages in `requirements.txt` need Python 3.11 or newer.

### 5. Deploy
Click **Create Web Service**. Render now installs the packages and starts the app. Watch the **Logs** tab; the first build takes a few minutes. When you see a line saying the service is live, open the URL at the top of the page:

```
https://placement-package-predictor.onrender.com
```

### Alternative: one-click with `render.yaml`
The repo already contains a Render Blueprint, so instead of steps 2–5:

1. Render dashboard → **New +** → **Blueprint**
2. Connect the `placement-predictor` repository
3. Render reads `render.yaml`, shows the service, and you click **Apply**

---

## Part 3: Updating the live site

Every `git push` to `main` redeploys automatically:

```bash
git add .
git commit -m "Describe your change"
git push
```

---

## Part 4: Put the live link in your README

Open `README.md`, find `https://YOUR-APP-NAME.onrender.com` and replace it with your real URL. Also replace `YOUR_USERNAME` in the clone command, then:

```bash
git add README.md
git commit -m "Add live demo link"
git push
```

Take screenshots of your live site and save them over the `assets/app_*.png` files to show the real fonts.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Build fails on `numpy` / `pandas` / `scikit-learn` | `PYTHON_VERSION` is missing or too old. Set it to `3.12.8` and redeploy. |
| Page shows "Bad Gateway" or never loads | The start command is wrong. It must contain `--server.port $PORT --server.address 0.0.0.0`. |
| First visit takes about a minute | Free instances sleep after 15 minutes without traffic and wake on the next visit. This is normal. |
| `FileNotFoundError: data/placement.csv` | The `data/` folder was not pushed. Run `git add data` and push again. |
| The CGPA box is blank or the page shows an empty frame | The `frontend/` folder was not pushed. Run `git add frontend` and push again. |
| Styles or fonts look different | Hard-refresh the browser with Ctrl+Shift+R. |
| Changes not showing | Check the **Events** tab on Render to confirm a new deploy started. |
