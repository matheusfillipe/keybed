const canvas = document.createElement("canvas");
canvas.style.position = "fixed";
canvas.style.inset = "0";
canvas.style.width = "100vw";
canvas.style.height = "100vh";
document.body.appendChild(canvas);

function paint(): void {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
}

paint();
window.addEventListener("resize", paint);
