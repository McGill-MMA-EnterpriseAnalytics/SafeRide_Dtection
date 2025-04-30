document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    const button = form.querySelector("button");
    const loader = document.getElementById("loader");

    form.addEventListener("submit", () => {
        button.disabled = true;
        button.innerText = "Processing...";
        if (loader) loader.style.display = "inline-block";
    });
});
