# Deploying the WhyBook Showcase

This guide outlines how to preview and publish the static WhyBook showcase page.

## 1. Local Preview

To preview the portfolio page locally without any build steps:

1. Open your terminal.
2. Navigate to the project root directory (`/home/asus/finetuning`).
3. Run Python's built-in HTTP server:
   ```bash
   python3 -m http.server 8080
   ```
4. Open [http://localhost:8080/index.html](http://localhost:8080/index.html) in your browser.

## 2. GitHub Pages Path

If you version-control this project using GitHub, GitHub Pages is the optimal hosting solution:

1. Push this repository to GitHub.
2. Navigate to your repository on GitHub.
3. Go to **Settings** > **Pages**.
4. Source: select **Deploy from a branch**.
5. Branch: select `main` (or your primary branch) and leave the folder as `/ (root)`.
6. Click **Save**. Your site will automatically build and deploy to `https://<your-username>.github.io/<repo-name>/`.

## 3. Hugging Face Space Static Hosting

To keep hosting directly alongside your models, you can host `index.html` inside a static Space:

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space Name (e.g., `whybook-showcase-page`) and choose **Static** as the Space SDK.
3. Clone your empty space locally:
   ```bash
   git clone https://huggingface.co/spaces/Stinger2311/<your-space-name>
   ```
4. Copy `index.html` into that Space folder.
5. Add, commit, and push:
   ```bash
   git add index.html
   git commit -m "Init showcase page"
   git push
   ```
6. The app will immediately be available at `https://huggingface.co/spaces/Stinger2311/<your-space-name>`.

> **Note:** Do not include `.env.local` or any private files in your commit when deploying.
