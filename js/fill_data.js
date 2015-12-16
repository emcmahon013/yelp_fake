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
        // .attr("class", "dl-horizontal")
      media_bosy_right.append("dt").append("h5").text("Type")
      media_bosy_right.append("dd").append("p").attr("class","lead").text(function(d){return(d["type"])})

      media_bosy_right.append("dt").append("h5").text("Probability")
      media_bosy_right.append("dd").append("p").attr("class","lead").text(function(d){return(d["probability"])})
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

  plot_deceptive_timeline = function(hotel_data, hotel_id){
    d3.select("#timeline-type-container svg").remove();
    var raw_positive_ts = hotel_data[hotel_id]["positive_ts"]
    var raw_negative_ts = hotel_data[hotel_id]["negative_ts"]
    
    var positive_ts = []
    var negative_ts = []
    raw_positive_ts["date"].forEach(function(date,i){
      positive_ts.push({"x": date, "y": raw_positive_ts["value"][i]});
      negative_ts.push({"x": date, "y": raw_negative_ts["value"][i]});
    })
    timeline = [{
                values: positive_ts,
                key: "Positive Deceptive",
                color: "#33a02c"
            },
            {
                values: negative_ts,
                key: "Negative Deceptive",
                color: "#e31a1c"
            }]
    return(timeline)
    
  }

  plot_stars_timeline = function(hotel_data, hotel_id){
    d3.select("#timeline-stars-container svg").remove();

    var raw_stars_ts = hotel_data[hotel_id]["star_ts"]
    
    var stars_ts = []
    var dates = [2007,2008,2009,2010,2011,2012,2013,2014]
    dates.forEach(function(date,i){
      stars_ts.push({"x": date, "y": raw_stars_ts["value"][i]});
    })
    st_timeline = [{
                values: stars_ts,
                key: "Adjusted Stars",
                color: "#FF9900"
            }]
    return(st_timeline)
    
  }

  timeline = plot_deceptive_timeline(review_data, hotel_default_id)
  st_timeline = plot_stars_timeline(review_data, hotel_default_id)
  var deceptive_chart;
    // console.log($("#timeline-type-container"))
    // $("#timeline-type-container").text("")
    // console.log($("#timeline-type-container"))

    nv.addGraph(function() {
        deceptive_chart = nv.models.lineChart()
            .options({
                transitionDuration: 300,
                useInteractiveGuideline: true
            })
        ;
        // chart sub-models (ie. xAxis, yAxis, etc) when accessed directly, return themselves, not the parent chart, so need to chain separately
        deceptive_chart.xAxis.tickValues([2007,2008,2009,2010,2011,2012,2013,2014]);
          
        deceptive_chart.yAxis.tickFormat(d3.format(',.2f'));

        deceptive_chart.showLegend(false);
        // data = sinAndCos();
        d3.select('#timeline-type-container').append('svg')
            .datum(timeline)
            .call(deceptive_chart);

        // d3.select('#timeline-stars-container').append('svg')
            // .datum(data)
            // .call(chart);
        nv.utils.windowResize(deceptive_chart.update);
        return deceptive_chart;
    });
    var stars_chart;
    // console.log($("#timeline-type-container"))
    // $("#timeline-type-container").text("")
    // console.log($("#timeline-type-container"))

    nv.addGraph(function() {
        stars_chart = nv.models.lineChart()
            .options({
                transitionDuration: 300,
                useInteractiveGuideline: true
            })
        ;
        // chart sub-models (ie. xAxis, yAxis, etc) when accessed directly, return themselves, not the parent chart, so need to chain separately
        stars_chart.xAxis.tickValues([2007,2008,2009,2010,2011,2012,2013,2014]);

        stars_chart.yAxis.tickFormat(d3.format(',.2f'));

        stars_chart.showLegend(false);
        // data = sinAndCos();
        d3.select('#timeline-stars-container').append('svg')
            .datum(st_timeline)
            .call(stars_chart);

        // d3.select('#timeline-stars-container').append('svg')
            // .datum(data)
            // .call(chart);
        nv.utils.windowResize(stars_chart.update);
        return stars_chart;
    });
    
  ///////////////////


  
  ///////////////////
  fill_hotels(hotel_default_id);
  fill_reviews(review_data, hotel_default_id)
  fill_rank(review_data, hotel_default_id)
  fill_hotel_metrics(review_data, hotel_default_id)
  fill_rank_table(hotels, hotel_default_id)
  // plot_deceptive_timeline(review_data, hotel_default_id)

  $(".hotel_card").click(function(e){
    $(".hotel_card").removeClass("active")
    $(this).addClass("active")
    id = $(this).data("id")
    fill_reviews(review_data, id)
    fill_rank(review_data, id)
    fill_hotel_metrics(review_data, id)
    timeline = plot_deceptive_timeline(review_data, id)
    st_timeline = plot_stars_timeline(review_data, id)

    var deceptive_chart;
      // console.log($("#timeline-type-container"))
      // $("#timeline-type-container").text("")
      // console.log($("#timeline-type-container"))

      nv.addGraph(function() {
          deceptive_chart = nv.models.lineChart()
              .options({
                  transitionDuration: 300,
                  useInteractiveGuideline: true
              })
          ;
          // chart sub-models (ie. xAxis, yAxis, etc) when accessed directly, return themselves, not the parent chart, so need to chain separately
          deceptive_chart.xAxis.tickValues([2007,2008,2009,2010,2011,2012,2013,2014]);
          deceptive_chart.yAxis.tickFormat(d3.format(',.2f'));
          deceptive_chart.showLegend(false);
          // data = sinAndCos();
          d3.select('#timeline-type-container').append('svg')
              .datum(timeline)
              .call(deceptive_chart);

          // d3.select('#timeline-stars-container').append('svg')
              // .datum(data)
              // .call(chart);
          nv.utils.windowResize(deceptive_chart.update);
          return deceptive_chart;
      });

    var stars_chart;
    // console.log($("#timeline-type-container"))
    // $("#timeline-type-container").text("")
    // console.log($("#timeline-type-container"))

    nv.addGraph(function() {
        stars_chart = nv.models.lineChart()
            .options({
                transitionDuration: 300,
                useInteractiveGuideline: true
            })
        ;
        // chart sub-models (ie. xAxis, yAxis, etc) when accessed directly, return themselves, not the parent chart, so need to chain separately
        stars_chart.xAxis.tickValues([2007,2008,2009,2010,2011,2012,2013,2014]);
        stars_chart.yAxis.tickFormat(d3.format(',.2f'));
        stars_chart.showLegend(false);
        // data = sinAndCos();
        d3.select('#timeline-stars-container').append('svg')
            .datum(st_timeline)
            .call(stars_chart);

        // d3.select('#timeline-stars-container').append('svg')
            // .datum(data)
            // .call(chart);
        nv.utils.windowResize(stars_chart.update);
        return stars_chart;
    });
    
      

    update_rank_table(hotels, id)
    e.preventDefault()
  });

})

