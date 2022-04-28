function create_graph(conv, trans) {
    ctx = document.getElementById("c").getContext("2d");
    ctx.lineWidth = 4;
    ctx.clearRect(0, 0, 600, 600);
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.beginPath();
    ctx.arc(300, 300, 200, 0, 2 * Math.PI);
    //ctx.moveTo(200, 0);
    //ctx.lineTo(400, 0);
    //ctx.moveTo(0,200);
    //ctx.lineTo(0,400);
    //ctx.strokeStyle = 'rgb(125,125,125)';
    ctx.stroke();
    for (var i = 0; i < 900; i += 25) {
        for (var j = 0; j < 900; j += 25) {

            rotation(ctx, i, j, conv, trans);
        }
    }

}


function rotation(ctx, x, y, conv, trans) {

    const r_max = 200;
    var headlen = 5;
    var dx = 300 - x;
    var dy = 300 - y;
    var angle = Math.atan2(dy, dx);
    var sina = Math.sin(angle)
    var cosa = Math.cos(angle)
    var distance = Math.sqrt(dx ** 2 + dy ** 2);
    let inner_r_coef = distance / r_max;
    let outer_r_coef = (r_max / distance) ** 2;
    var unit = 10;


    if (distance < r_max) {
        var coef = inner_r_coef;
    } else {
        var coef = outer_r_coef;
    }

    var fmx = x + unit * coef * sina;
    var tox = x - unit * coef * sina + trans;
    var fmy = y - unit * coef * cosa;
    var toy = y + unit * coef * cosa;

    // create proportional vector length option    
    fmx = fmx - conv * coef * cosa;
    tox = tox + conv * coef * cosa;
    fmy = fmy - conv * coef * sina;
    toy = toy + conv * coef * sina;

    // new_angle is orientation of vector instead of position from origin
    var full_mag = Math.sqrt((toy - fmy) ** 2 + (tox - fmx) ** 2);
    var new_angle = Math.atan2(toy - fmy, tox - fmx)

    // create uniform vector length option
    // this requires both magnitude and direction from the full magnitude vector
    //
    fmx = x - (unit * Math.cos(new_angle));
    tox = x + (unit * Math.cos(new_angle));
    fmy = y - (unit * Math.sin(new_angle));
    toy = y + (unit * Math.sin(new_angle));


    ctx.lineWidth = 1
    if (full_mag > 33) {
        ctx.lineWidth = 3
        var stroke_color = 'rgb(128,0,128';
    } else if (full_mag > 27) {
        ctx.lineWidth = 2.5
        var stroke_color = 'rgb(225, 87, 51)';
    } else if (full_mag > 24) {
        ctx.lineWidth = 2
        var stroke_color = 'rgba(200, 170, 0.9)';
    } else if (full_mag > 22) {
        ctx.lineWidth = 1.5
        var stroke_color = 'rgba(200, 170, 0.8)';
    } else if (full_mag > 20) {
        var stroke_color = 'rgba(0, 125, 0,0.8)';
    } else if (full_mag > 14) {
        var stroke_color = 'rgba(0, 125, 0,0.6)';
    } else if (full_mag > 12) {
        var stroke_color = 'rgba(0, 0,0,0.6)';
    } else if (full_mag > 8) {
        var stroke_color = 'rgba(0, 0,0,0.5)';
    } else if (full_mag > 4) {
        var stroke_color = 'rgba(0, 0,0,0.4)';
    } else {
        var stroke_color = 'rgba(0, 0,0,0.2)';
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

    var conv = 10 - parseFloat(id.slice(1, 3));
    var trans = parseFloat(id.slice(3, 5));
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
        //console.log(e.target.id)
        get_values(e.target.id);
    })
});

squares.forEach(square => {
    square.addEventListener('mouseleave', e => {
        document.getElementById(e.target.id).classList.remove('highlight');
    })
});