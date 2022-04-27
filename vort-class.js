let ctx = document.getElementById("c").getContext("2d");

class Vector {
    constructor(id) {
        this.id = id;
        this.trans = parseFloat(this.id.slice(1, 2)) * 2;
        this.conv = parseFloat(this.id.slice(2, 3)) * 2;
        this.x = 0;
        this.y = 0;

        console.log(this.conv)
        this.convergence = 'Convergence To Rotation Ratio: ' + this.conv.toString() + '\/10';
        this.translation = 'Translation To Rotation Ratio: ' + this.trans.toString() + '\/10';
        document.getElementById('conv').innerText = this.convergence;
        document.getElementById('trans').innerText = this.translation;

        this.create_graph();
    }

    create_graph() {

        //self.ctx.clearRect(0, 0, 600, 600);
        ctx.beginPath()
        for (this.x = 0; this.x < 900; this.x += 25) {
            for (this.y = 0; this.y < 900; this.y += 25) {
                this.rotation(ctx, this.x, this.y, this.conv, this.trans);
                ctx.stroke();
            }
        }
        
        }

    rotation() {

        const r_max = 200;
        var hdlen = 3;
        var dx = 300 - this.x;
        var dy = 300 - this.y;

        var angle = Math.atan2(dy, dx);
        var sina = Math.sin(angle)
        var cosa = Math.cos(angle)

        var distance = Math.sqrt(dx ** 2 + dy ** 2);
        //console.log(distance);

        let inner_r_coef = distance / r_max;
        let outer_r_coef = (r_max / distance) ** 2;
        const unit = 10;


        if (distance < r_max) {
            var coef = inner_r_coef;
        } else {
            var coef = outer_r_coef;
        }

        var fmx = this.x + unit * coef * sina;
        var tox = this.x - unit * coef * sina + self.trans;
        var fmy = this.y - unit * coef * cosa;
        var toy = this.y + unit * coef * cosa;

        fmx = fmx - this.conv * coef * cosa;
        tox = tox + this.conv * coef * cosa;
        fmy = fmy - this.conv * coef * sina;
        toy = toy + this.conv * coef * sina;


        this.final_mag = Math.sqrt((toy - fmy) ** 2 + (tox - fmx) ** 2);
        console.log(this.final_mag);

        //if (self.final_mag > 15) {
        //    ctx.strokeColor = 'rgb(250,100,125)';
        //} else {
        //    self.strokeColor = 'rgb(100,100, 125)';
        //};


        let color_calc = Math.floor(255 * this.final_mag / 55);
        let color_calc_minus = 255 - color_calc;
        // new_angle is orientation of vector instead of position from origin
        var new_angle = Math.atan2(this.toy - this.fmy, this.tox - this.fmx)

        ctx.moveTo(this.fmx, this.fmy);
        ctx.lineTo(this.tox, this.toy);
        ctx.moveTo(this.tox, this.toy);
        ctx.lineTo(this.tox - hdlen * Math.cos(new_angle - Math.PI / 6), this.toy - hdlen * Math.sin(
            new_angle -
            Math
            .PI / 6));
        ctx.moveTo(this.tox, this.toy);
        ctx.lineTo(this.tox - hdlen * Math.cos(new_angle + Math.PI / 6), this.toy - hdlen * Math.sin(
            new_angle +
            Math
            .PI / 6));
        //ctx.strokeColor = `rgb(${color_calc},${color_calc_minus},125)`;
        //console.log(ctx.strokeColor);
        
    }

    plot() {

    }


}

const squares = document.querySelectorAll(".r");

squares.forEach(square => {
    square.addEventListener('mouseenter', e => {

        document.getElementById(e.target.id).classList.add('highlight');
        let el = document.getElementById(e.target.id);
        let newVector = new Vector(e.target.id);
    })
});

squares.forEach(square => {
    square.addEventListener('mouseleave', e => {
        document.getElementById(e.target.id).classList.remove('highlight');
    })
});
