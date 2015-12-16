$(function() {
  
  var review_data = (function () {
      var json = null;
      $.ajax({
          'async': false,
          'global': false,
          'url': "data/reviews.json",
          'dataType': "json",
          'success': function (data) {
              json = data;
          }
      });
      return json;
  })(); 

  var hotels = []

  for(var key in review_data){
    hotels.push({"id":key, 
      "star":review_data[key]["star"],
      "rounded_star":review_data[key]["rounded_star"],
      "image_url": review_data[key]["image_url"],
      "city": review_data[key]["city"],
      "name": review_data[key]["name"],
      "rank": review_data[key]["rank"],
      "positive": review_data[key]["positive"],
      "negative": review_data[key]["negative"]
    });
  }
  console.log(hotels)
  hotels.sort(function(x, y){
    return d3.descending(x.rank, y.rank);
  })
  var hotel_default_id = hotels[0].id
  fill_hotels = function(hotel_active){
    var hotel_cards = d3.select("#hotels-list").selectAll("a")
      .data(hotels).enter()
      .append("a")
        .attr("class", "list-group-item hotel_card")
        .attr("id", function(d){return("hotel_card_" + d["id"])})
        .attr("data-id", function(d){return(d["id"])})
        .attr("href", "#")
      .append("div")
        .attr("class", "row")
      
    hotel_cards.append("div")
      .attr("class", "col-md-6")
      .append("img")
      // .attr("src", function(d){return(d["image_url"])})
      .attr("src", function(d){return("./img/hotels/"+d["id"] + ".jpg")})
      .attr("height", 80)
      .attr("width", 80)

    var hotel_right_inner_list = hotel_cards.append("div")
      .attr("class", "col-md-6")
      .style("padding-left", "0px")
      .append("ul")
      .attr("class", "list-unstyled")

    hotel_right_inner_list.append("li")
      .append("h4")
      .text(function(d){return(d["name"])})

    hotel_right_inner_list.append("li")
      .text(function(d){return(d["city"])})

    hotel_right_inner_list.append("img")
      .attr("src", function(d){
        return("./img/"+ d.rounded_star +"_stars.svg.png")
      })
      .attr("height", 20)

    $("#hotel_card_"+ hotel_active).addClass("active");
  }

  fill_reviews = function(data,hotel_id){

    reviews = data[hotel_id]["reviews"]
    
    $("#reviews-container").fadeOut(function() {
      $(this).text("")
      var review_card = d3.select("#reviews-container").selectAll("div")
        .data(reviews).enter()
        .append("div")
        .attr("id", function(d){return("review_card_" + d["id"])})
        .attr("class", function(d){return "media review-card " + d["type"].replace(/\s+/g, '-').toLowerCase()+"-card"})
       
      var media_left = review_card.append("div")
        .attr("class", "media-left")

      media_left.append("img")
        .attr("class", "media-object")
        .attr("src", "./img/contact-outline.png")
        .attr("width", "50")
        .attr("height", "50")


      var media_body = review_card.append("div")
        .attr("class", "media-body")

      var media_body_list = media_body.append("ul")
        .attr("class", "list-inline")

      media_body_list.append("li").append("h4").text(function(d){return(d["name"])})

      media_body_list.append("li")
        .append("img")
        .attr("src", function(d){
          return("./img/"+ d.star +"_stars.svg.png")
        })
        .attr("height", 20)
      
      media_body_list.append("li")
        .text(function(d){return(d["date"])})

      media_body.append("p")
        .text(function(d){return(d["review"])})

      var media_bosy_right = review_card.append("div")
        .attr("class", "media-right")
        .append("dl")
        .attr("class", "dl-horizontal")
      media_bosy_right.append("dt").text("Type")
      media_bosy_right.append("dd").text(function(d){return(d["type"])})

      media_bosy_right.append("dt").text("Probability")
      media_bosy_right.append("dd").text(function(d){return(d["probability"])})
  }).fadeIn();
  }

  fill_rank_table = function(hotel_data, hotel_id){
    var table_container = d3.select("#ranking-table")

    var table_head = table_container.append("thead").append("tr")
    
    table_head.append("th").text("#")
    table_head.append("th").text("Hotel")

    var table_body = table_container.append("tbody").selectAll("tr")
        .data(hotel_data).enter()
        .append("tr")
        .attr("class", function(d){
          if(d.id == hotel_id)
            return "highlight rank"
          else
            return "rank"
        })
        .attr("id",function(d){return("rank_"+d.id)})

    table_body.append("td").text(function(d,i){return(i+1)})  
    table_body.append("td").text(function(d){return(d["name"])})
  }

  update_rank_table = function(hotel_data, hotel_id){
    $(".rank").removeClass("highlight");
    $('#rank_'+hotel_id).addClass('highlight');
  }

  fill_rank = function(hotel_data, hotel_id){
    var rank = hotel_data[hotel_id]["rank"]

    $("#ranking-content").fadeOut(function() {
      $(this).text("Top "+ rank + "%")
    }).fadeIn();
  }

  fill_hotel_metrics = function(hotel_data, hotel_id){
    var positive = hotel_data[hotel_id]["positive"]
    var negative = hotel_data[hotel_id]["negative"]
    var adjusted_stars = hotel_data[hotel_id]["star"]

    $("#nd-text").fadeOut(function() {
      $(this).text(negative)
    }).fadeIn();

    $("#pd-text").fadeOut(function() {
      $(this).text(positive)
    }).fadeIn();

    $("#adj-stars-text").fadeOut(function() {
      $(this).text(adjusted_stars)
    }).fadeIn();
  }

  ///////////////////

  var chart;
    var data;
    var randomizeFillOpacity = function() {
        var rand = Math.random(0,1);
        for (var i = 0; i < 100; i++) { // modify sine amplitude
            data[4].values[i].y = Math.sin(i/(5 + rand)) * .4 * rand - .25;
        }
        data[4].fillOpacity = rand;
        chart.update();
    };
    nv.addGraph(function() {
        chart = nv.models.lineChart()
            .options({
                transitionDuration: 300,
                useInteractiveGuideline: true
            })
        ;
        // chart sub-models (ie. xAxis, yAxis, etc) when accessed directly, return themselves, not the parent chart, so need to chain separately
        chart.xAxis
            // .axisLabel("Time (s)")
            .tickFormat(d3.format(',.1f'))
            .staggerLabels(true)
        ;
        chart.yAxis
            // .axisLabel('Voltage (v)')
            .tickFormat(function(d) {
                if (d == null) {
                    return 'N/A';
                }
                return d3.format(',.2f')(d);
            })
        ;
        chart.showLegend(false);
        data = sinAndCos();
        d3.select('#timeline-type-container').append('svg')
            .datum(data)
            .call(chart);

        d3.select('#timeline-stars-container').append('svg')
            .datum(data)
            .call(chart);
        nv.utils.windowResize(chart.update);
        return chart;
    });
    function sinAndCos() {
        var sin = [],
            sin2 = [],
            cos = [],
            rand = [],
            rand2 = []
            ;
        for (var i = 0; i < 100; i++) {
            sin.push({x: i, y: i % 10 == 5 ? null : Math.sin(i/10) }); //the nulls are to show how defined works
            sin2.push({x: i, y: Math.sin(i/5) * 0.4 - 0.25});
            cos.push({x: i, y: .5 * Math.cos(i/10)});
            rand.push({x:i, y: Math.random() / 10});
            rand2.push({x: i, y: Math.cos(i/10) + Math.random() / 10 })
        }
        return [
            // {
            //     area: true,
            //     values: sin,
            //     key: "Sine Wave",
            //     color: "#ff7f0e",
            //     strokeWidth: 4,
            //     classed: 'dashed'
            // },
            {
                values: cos,
                key: "Cosine Wave",
                color: "#2ca02c"
            },
            {
                values: rand,
                key: "Random Points",
                color: "#2222ff"
            },
            {
                values: rand2,
                key: "Random Cosine",
                color: "#667711",
                strokeWidth: 3.5
            }
            // {
            //     area: true,
            //     values: sin2,
            //     key: "Fill opacity",
            //     color: "#EF9CFB",
            //     fillOpacity: .1
            // }
        ];
    }
  ///////////////////
  fill_hotels(hotel_default_id);
  fill_reviews(review_data, hotel_default_id)
  fill_rank(review_data, hotel_default_id)
  fill_hotel_metrics(review_data, hotel_default_id)
  fill_rank_table(hotels, hotel_default_id)

  $(".hotel_card").click(function(e){
    $(".hotel_card").removeClass("active")
    $(this).addClass("active")
    id = $(this).data("id")
    fill_reviews(review_data, id)
    fill_rank(review_data, id)
    fill_hotel_metrics(review_data, id)

    update_rank_table(hotels, id)
    e.preventDefault()
  });

})

