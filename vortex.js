function create_graph(conv, trans) {
    ctx = document.getElementById("c").getContext("2d");
    ctx.clearRect(0, 0, 600, 600);

    for (var i = 0; i < 900; i += 25) {
        for (var j = 0; j < 900; j += 25) {

            rotation(ctx, i, j, conv, trans);
        }
    }

}


function rotation(ctx, x, y, conv, trans) {

    const r_max = 200;
    var headlen = 3;
    var dx = 300 - x;
    var dy = 300 - y;
    var angle = Math.atan2(dy, dx);
    var sina = Math.sin(angle)
    var cosa = Math.cos(angle)
    var distance = Math.sqrt(dx ** 2 + dy ** 2);
    let inner_r_coef = distance / r_max;
    let outer_r_coef = (r_max / distance) ** 2;
    var unit = 4;


    if (distance < r_max) {
        var coef = inner_r_coef;
    } else {
        var coef = outer_r_coef;
    }

    var fmx = x + unit * coef * sina;
    var tox = x - unit * coef * sina + trans;
    var fmy = y - unit * coef * cosa;
    var toy = y + unit * coef * cosa;

    fmx = fmx - conv * coef * cosa;
    tox = tox + conv * coef * cosa;
    fmy = fmy - conv * coef * sina;
    toy = toy + conv * coef * sina;

    // new_angle is orientatin of vector instead of position from origin
    var final_mag = Math.sqrt((toy - fmy) ** 2 + (tox - fmx) ** 2);
    console.log(final_mag);

    //let color_calc = Math.floor(2550 * final_mag);
    //let color_calc_minus = 255 - (color_calc*10);
    var new_angle = Math.atan2(toy - fmy, tox - fmx)

    if (final_mag > 50 ) {
        var stroke_color = 'rgb(128,0,128';
    } else if (final_mag > 45 ) {
        var stroke_color = 'rgb(175, 0, 0)';
    } else if (final_mag > 25 ) {
        var stroke_color = 'rgb(25, 25, 175)';
    } else if (final_mag > 20 ) {
        var stroke_color = 'rgb(25, 25, 200)';
    } else if (final_mag > 12 ) {
        var stroke_color = 'rgb(75, 75, 75)';
    } else if (final_mag > 8 ) {
        var stroke_color = ' rgb(100, 100, 100)';
    } else if (final_mag > 5 ) {
        var stroke_color = 'rgb(150,150,150)';
    } else {
        var stroke_color = 'rgb(200, 200, 200)';
    }   
  
   
    
    
    
    
    
    
    


    ctx.beginPath();
    ctx.moveTo(fmx, fmy);
    ctx.lineTo(tox, toy);
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox - headlen * Math.cos(new_angle - Math.PI / 6), toy - headlen * Math.sin(new_angle - Math
        .PI / 6));
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox - headlen * Math.cos(new_angle + Math.PI / 6), toy - headlen * Math.sin(new_angle + Math
        .PI / 6));
        ctx.strokeStyle = stroke_color;
        ctx.stroke();
    //function plot_arrow()
//return;
}


function get_values(id) {

    var trans = parseFloat(id.slice(1, 3)) * 2;
    var conv = parseFloat(id.slice(3, 5)) * 2;
    var convergence = 'Max Convergence: ' + conv.toString();
    var translation = 'Max Translation: ' + trans.toString();
    document.getElementById('conv').innerText = convergence;
    document.getElementById('trans').innerText = translation;

    create_graph(conv, trans);

};

const squares = document.querySelectorAll(".r");

squares.forEach(square => {
    square.addEventListener('mouseenter', e => {

        document.getElementById(e.target.id).classList.add('highlight');
        let el = document.getElementById(e.target.id);
        console.log(e.target.id)
        get_values(e.target.id);
    })
});

squares.forEach(square => {
    square.addEventListener('mouseleave', e => {
        document.getElementById(e.target.id).classList.remove('highlight');
    })
});