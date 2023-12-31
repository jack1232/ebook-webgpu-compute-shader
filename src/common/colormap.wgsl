fn colormapData(colormapId:u32) -> array<vec3f, 11>{
    var colors:array<vec3f, 11>;
    if(colormapId == 1u) { // hsv
        colors = array<vec3f, 11>(
            vec3(1.0,0.0,0.0),
            vec3(1.0,0.5,0.0),
            vec3(0.97,1.0,0.01),
            vec3(0.0,0.99,0.04),
            vec3(0.0,0.98,0.52),
            vec3(0.0,0.98,1.0),
            vec3(0.01,0.49,1.0),
            vec3(0.03,0.0,0.99),
            vec3(1.0,0.0,0.96),
            vec3(1.0,0.0,0.49),
            vec3(1.0,0.0,0.02));
    } else if(colormapId == 2u) { // hot
        colors = array<vec3f, 11>(
            vec3(0.0,0.0,0.0),
            vec3(0.3,0.0,0.0),
            vec3(0.6,0.0,0.0),
            vec3(0.9,0.0,0.0),
            vec3(0.93,0.0,0.0),
            vec3(0.97,0.55,0.0),
            vec3(1.0,0.82,0.0),
            vec3(1.0,0.87,0.25),
            vec3(1.0,0.91,0.5),
            vec3(1.0,0.96,0.75),
            vec3(1.0,1.0,1.0));
    } else if(colormapId == 3u) { // cool
        colors = array<vec3f, 11>(
            vec3(0.49,0.0,0.7),
            vec3(0.45,0.0,0.85),
            vec3(0.42,0.15,0.89),
            vec3(0.38,0.29,0.93),
            vec3(0.27,0.57,0.910),
            vec3(0.0,0.8,0.77),
            vec3(0.0,0.97,0.57),
            vec3(0.0,0.98,0.46),
            vec3(0.0,1.0,0.35),
            vec3(0.16,1.0,0.03),
            vec3(0.58,1.0,0.0));
    } else if(colormapId == 4u) { // spring
        colors = array<vec3f, 11>(
            vec3(1.0,0.0,1.0),
            vec3(1.0,0.1, 0.9),
            vec3(1.0,0.2,0.8),
            vec3(1.0,0.3,0.7),
            vec3(1.0,0.4,0.6),
            vec3(1.0,0.5,0.5),
            vec3(1.0,0.6,0.4),
            vec3(1.0,0.7,0.3),
            vec3(1.0,0.8,0.2),
            vec3(1.0,0.9,0.1),
            vec3(1.0,1.0,0.0));
    } else if(colormapId == 5u) { // summer
        colors = array<vec3f, 11>(
            vec3(0.0,0.5,0.4),
            vec3(0.1,0.55,0.4),
            vec3(0.2,0.6,0.4),
            vec3(0.3,0.65,0.4),
            vec3(0.4,0.7,0.4),
            vec3(0.5,0.75,0.4),
            vec3(0.6,0.8,0.4),
            vec3(0.7,0.85,0.4),
            vec3(0.8,0.9,0.4),
            vec3(0.9,0.95,0.4),
            vec3(1.0,1.0,0.4));
    } else if(colormapId == 6u) { // autumn
        colors = array<vec3f, 11>(
            vec3(1.0,0.0,0.0),
            vec3(1.0,0.1,0.0),
            vec3(1.0,0.2,0.0),
            vec3(1.0,0.3,0.0),
            vec3(1.0,0.4,0.0),
            vec3(1.0,0.5,0.0),
            vec3(1.0,0.6,0.0),
            vec3(1.0,0.7,0.0),
            vec3(1.0,0.8,0.0),
            vec3(1.0,0.9,0.0),
            vec3(1.0,1.0,0.0));
    } else if(colormapId == 7u) { // winter
        colors = array<vec3f, 11>(
            vec3(0.0,0.0,1.0),
            vec3(0.0,0.1,0.95),
            vec3(0.0,0.2,0.9),
            vec3(0.0,0.3,0.85),
            vec3(0.0,0.4,0.8),
            vec3(0.0,0.5,0.75),
            vec3(0.0,0.6,0.7),
            vec3(0.0,0.7,0.65),
            vec3(0.0,0.8,0.6),
            vec3(0.0,0.9,0.55),
            vec3(0.0,1.0,0.5));
    } else if(colormapId == 8u) { // bone
        colors = array<vec3f, 11>(
            vec3(0.0,0.0,0.0),
            vec3(0.08,0.08,0.11),
            vec3(0.16,0.16,0.23),
            vec3(0.25,0.25,0.34),
            vec3(0.33,0.33,0.45),
            vec3(0.41,0.44,0.54),
            vec3(0.5,0.56,0.62),
            vec3(0.58,0.67,0.7),
            vec3(0.66,0.78,0.78),
            vec3(0.83,0.89,0.89),
            vec3(1.0,1.0,1.0));
    } else if(colormapId == 9u) { // cooper
        colors = array<vec3f, 11>(
            vec3(0.0,0.0,0.0),
            vec3(0.13,0.08,0.05),
            vec3(0.25,0.16,0.1),
            vec3(0.38,0.24,0.15),
            vec3(0.5,0.31,0.2),
            vec3(0.62,0.39,0.25),
            vec3(0.75,0.47,0.3),
            vec3(0.87,0.55,0.35),
            vec3(1.0,0.63,0.4),
            vec3(1.0,0.71,0.45),
            vec3(1.0,0.78,0.5));
    } else if(colormapId == 10u) { // greys
        colors = array<vec3f, 11>(
            vec3(0.0,0.0,0.0),
            vec3(0.1,0.1,0.1),
            vec3(0.2,0.2,0.2),
            vec3(0.3,0.3,0.3),
            vec3(0.4,0.4,0.4),
            vec3(0.5,0.5,0.5),
            vec3(0.6,0.6,0.6),
            vec3(0.7,0.7,0.7),
            vec3(0.8,0.8,0.8),
            vec3(0.9,0.9,0.9),
            vec3(1.0,1.0,1.0));
    } else if(colormapId == 11u) { // rainbow
        colors = array<vec3f, 11>(
            vec3(0.588, 0.000, 0.353),
            vec3(0.118, 0.000, 0.698),
            vec3(0.000, 0.059, 0.914),
            vec3(0.000, 0.297, 1.000),
            vec3(0.035, 0.677, 0.918),
            vec3(0.173, 1.000, 0.588),
            vec3(0.508, 1.000, 0.118),
            vec3(0.837, 0.951, 0.000),
            vec3(1.000, 0.725, 0.000),
            vec3(1.000, 0.348, 0.000),
            vec3(1.000, 0.000, 0.000));
    } else if(colormapId == 12u) { // rainbow_soft
        colors = array<vec3f, 11>(
            vec3(0.490, 0.000, 0.702),
            vec3(0.780, 0.000, 0.706),
            vec3(1.000, 0.000, 0.475),
            vec3(1.000, 0.424, 0.000),
            vec3(0.871, 0.761, 0.000),
            vec3(0.588, 1.000, 0.000),
            vec3(0.000, 1.000, 0.216),
            vec3(0.000, 0.965, 0.588),
            vec3(0.196, 0.655, 0.871),
            vec3(0.404, 0.200, 0.922),
            vec3(0.486, 0.000, 0.729));
    } else if(colormapId == 13u) { // white
        colors = array<vec3f, 11>(
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0));
    } else if(colormapId == 14u) { // black
        colors = array<vec3f, 11>(
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0));
    } else if(colormapId == 15u) { // red
        colors = array<vec3f, 11>(
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0));
    } else if(colormapId == 16u) { // green
        colors = array<vec3f, 11>(
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0));
    } else if(colormapId == 17u) { // blue
        colors = array<vec3f, 11>(
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0));
    } else if(colormapId == 18u) { // yellow
        colors = array<vec3f, 11>(
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0));
    } else if(colormapId == 19u) { // cyan
        colors = array<vec3f, 11>(
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0),
            vec3(0.0, 1.0, 1.0));
    } else if(colormapId == 20u) { // fuchsia
        colors = array<vec3f, 11>(
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0),
            vec3(1.0, 0.0, 1.0));
    } else if(colormapId == 21u) { // terrain
        colors = array<vec3f, 11>(
            vec3(0.1765,0.2471,0.6471),
            vec3(0.0392,0.5176,0.9176),
            vec3(0.0000,0.7451,0.5725),
            vec3(0.3098,0.8627,0.4588),
            vec3(0.7098,0.9451,0.5451),
            vec3(0.9686,0.9608,0.5843),
            vec3(0.7686,0.7059,0.4784),
            vec3(0.5451,0.4196,0.3529),
            vec3(0.6196,0.5098,0.4863),
            vec3(0.7765,0.7137,0.7020),
            vec3(0.9490,0.9333,0.9333));
    } else if(colormapId == 22u) { // ocean
        colors = array<vec3f, 11>(
            vec3(0.0000,0.4627,0.0275),
            vec3(0.0000,0.3216,0.1176),
            vec3(0.0000,0.1686,0.2196),
            vec3(0.0000,0.0392,0.3098),
            vec3(0.0000,0.0902,0.3961),
            vec3(0.0000,0.2275,0.4863),
            vec3(0.0000,0.3804,0.5843),
            vec3(0.0510,0.5255,0.6863),
            vec3(0.3137,0.6549,0.7686),
            vec3(0.5922,0.7961,0.8627),
            vec3(0.9020,0.9490,0.9647));
    } else if (colormapId == 0u) { // jet (default) 0, 
        colors = array<vec3f, 11>(
            vec3(0.0,0.0,0.51),
            vec3(0.0,0.24,0.67),
            vec3(0.01,0.49,0.78),
            vec3(0.01,0.75,0.89),
            vec3(0.02,1.0,1.0),
            vec3(0.51,1.0,0.5),
            vec3(1.0,1.0,0.0),
            vec3(0.99,0.67,0.0),
            vec3(0.99,0.33,0.0),
            vec3(0.98,0.0,0.0),
            vec3(0.5,0.0,0.0));
    }
    return colors;
}

fn colorLerp(colormapId:u32, tmin:f32, tmax:f32, t:f32, colormapReverse:u32) -> vec3f{
    var t1 = t;
    if (t1 < tmin) {t1 = tmin;}
    if (t1 > tmax) {t1 = tmax;}
    var tn = (t1-tmin)/(tmax-tmin);

    if(colormapReverse >= 1u) {tn = 1.0 - tn;}

    var colors = colormapData(colormapId);
    var idx = u32(floor(10.0*tn));
    var color = vec3(0.0,0.0,0.0);

    if(f32(idx) == 10.0*tn) {
        color = colors[idx];
    } else {
        var tn1 = (tn - 0.1*f32(idx))*10.0;
        var a = colors[idx];
        var b = colors[idx+1u];
        color.x = a.x + (b.x - a.x)*tn1;
        color.y = a.y + (b.y - a.y)*tn1;
        color.z = a.z + (b.z - a.z)*tn1;
    }
    return color;
}