const username = "JosueMonte";

const exclude = [
  "JosueMonte.github.io"
];

async function loadRepos() {
  const container = document.getElementById("repo-list");

  try {
    const response = await fetch(`https://api.github.com/users/${username}/repos`);
    const repos = await response.json();

    const filtered = repos.filter(repo =>
      !repo.fork &&
      !exclude.includes(repo.name)
    );

    filtered.forEach(repo => {
      const col = document.createElement("div");
      col.className = "col-md-4 mb-4";

      col.innerHTML = `
        <div class="card h-100">
          <div class="card-body">
            <h4 class="card-title text-primary">${repo.name.replace(/-/g, " ")}</h4>
            <p class="card-text">${repo.description || "Repositorio sin descripción."}</p>
            <a href="${repo.html_url}" target="_blank" class="btn btn-outline-primary mt-2">
              <i class="fab fa-github"></i> Ir al repo
            </a>

            ${
              repo.homepage
                ? `<br><a href="${repo.homepage}" target="_blank" class="btn btn-primary mt-2">Demo / App</a>`
                : ""
            }
          </div>
        </div>
      `;

      container.appendChild(col);
    });

  } catch (error) {
    console.error("Error cargando repositorios:", error);
  }
}

loadRepos();

/* ---------- MEDIUM BLOG FEED ---------- */

const mediumUsername = "josue.monte";   // tu usuario de Medium

async function loadMediumPosts() {
  const mediumContainer = document.getElementById("blog-container");

  try {
    const res = await fetch(
      `https://api.rss2json.com/v1/api.json?rss_url=https://medium.com/feed/@${mediumUsername}`
    );
    const data = await res.json();

    const posts = data.items.slice(0, 3);

    posts.forEach(post => {
      const col = document.createElement("div");
      col.className = "col-md-4 mb-4";

      col.innerHTML = `
        <div class="card h-100">
          <div class="card-body">
            <h4 class="card-title text-primary">${post.title}</h4>
            <p class="card-text">${post.description.replace(/<[^>]+>/g, "").substring(0, 150)}...</p>
            <a href="${post.link}" target="_blank" class="btn btn-outline-primary mt-2">
              Leer en Medium
            </a>
          </div>
        </div>
      `;

      mediumContainer.appendChild(col);
    });

  } catch (e) {
    console.error("Error cargando Medium:", e);
  }
}

loadMediumPosts();
