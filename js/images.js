d3.select(window).on("load", console.log("IMAGES"));





function reply_click(clicked_id) {
  d3.select('#exampleModalLongTitle').text("");
  d3.select('#esperado').text("");
  d3.select('#modelo').text("");
  d3.select(".loading").style("display", "block");

  


  d3.json("http://localhost:4000/analize/" + clicked_id).then((d) => {
    console.log(d[0])

    d3.select('#exampleModalLongTitle').text("Nombre: " + d[0][2]);
    d3.select('#esperado').text("Resultado Esperado: " + d[0][4]);
    d3.select('#modelo').text("Resultado del Modelo: " + d[0][5]);
    d3.selectAll(".loading").style("display", "none")
  });

}

function reset() {

  d3.select('#exampleModalLongTitle').text("");
  d3.select('#esperado').text("");
  d3.select('#modelo').text("");
  d3.select(".loading").style("display", "block");

}

