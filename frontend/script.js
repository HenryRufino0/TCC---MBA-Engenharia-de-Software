const apiBase = "http://127.0.0.1:8000";
let historico = [];

document.getElementById("taskForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const tarefa = {
    nome: document.getElementById("nomeTarefa").value,
    frequencia: parseInt(document.getElementById("frequencia").value),
    tempo: parseInt(document.getElementById("tempo").value),
    complexidade: document.getElementById("complexidade").value,
    importancia: document.getElementById("importancia").value,
    urgencia: document.getElementById("urgencia").value,
    ferramentas: parseInt(document.getElementById("ferramentas").value),
    volume: parseInt(document.getElementById("volume").value),
    colaboradores: parseInt(document.getElementById("colaboradores").value)
  };

  try {
    const res = await fetch(`${apiBase}/avaliar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tarefa)
    });

    const data = await res.json();
    const resultadoDiv = document.getElementById("resultado");

    if (res.ok) {
      resultadoDiv.textContent = data.automatizar === "sim"
        ? "✅ Essa tarefa deve ser automatizada!"
        : "❌ Essa tarefa NÃO deve ser automatizada.";
      resultadoDiv.className = "resultado " + data.automatizar;

      historico.push({ ...tarefa, resultado: data.automatizar });
      atualizarHistorico();
    } else {
      resultadoDiv.textContent = "Erro ao avaliar.";
      resultadoDiv.className = "resultado nao";
    }
  } catch (err) {
    console.error(err);
    document.getElementById("resultado").textContent = "Erro na requisição.";
  }
});

function atualizarHistorico() {
  const historicoDiv = document.getElementById("historico");
  historicoDiv.innerHTML = historico.map(
    (h, i) =>
      `<p><b>${i + 1}. ${h.nome}</b> → ${h.resultado.toUpperCase()}</p>`
  ).join("");
}

function exportarPDF() {
  if (historico.length === 0) {
    alert("Nenhum resultado para exportar.");
    return;
  }

  
  let conteudo = `<h2 style="text-align:center; font-family:Segoe UI">Resultados</h2>`;
  conteudo += "<ul style='font-family:Segoe UI; font-size:14px;'>";
  historico.forEach((h, i) => {
    conteudo += `<li><b>${i + 1}. ${h.nome}</b> → ${h.resultado.toUpperCase()}</li>`;
  });
  conteudo += "</ul>";

  const elementoTemporario = document.createElement("div");
  elementoTemporario.innerHTML = conteudo;

  
  html2pdf().from(elementoTemporario).set({
    margin: 10,
    filename: "resultados.pdf",
    html2canvas: { scale: 2 },
    jsPDF: { unit: "mm", format: "a4", orientation: "portrait" }
  }).save();
}

function toggleMode() {
  document.body.dataset.theme =
    document.body.dataset.theme === "dark" ? "light" : "dark";
}
