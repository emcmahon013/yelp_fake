$(function() {
  var hotel_default_id = "amalfi"
  var data = (function () {
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

  for(var key in data){
    hotels.push({"id":key, 
      "stars":data[key]["stars"],
      "image_url": data[key]["image_url"],
      "city": data[key]["city"],
      "name": data[key]["name"]
    });
  }
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
      .attr("src", function(d){return(d["image_url"])})
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
        return("./img/"+ d.stars +"_stars.svg.png")
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
        .attr("class", "media")
       
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
          return("./img/"+ d.stars +"_stars.svg.png")
        })
        .attr("height", 20)
      
      media_body_list.append("li")
        .text(function(d){return(d["date"])})

      media_body.append("p")
        .text(function(d){return(d["review"])})

      var media_bosy_right = review_card.append("div")
        .attr("class", "media-right")
        .append("dl")
      media_bosy_right.append("dt").text("Type")
      media_bosy_right.append("dd").text(function(d){return(d["type"])})

      media_bosy_right.append("dt").text("Probability")
      media_bosy_right.append("dd").text(function(d){return(d["probability"])})
  }).fadeIn();
  }

  fill_rank = function(data, hotel_id){
    var rank = data[hotel_id]["rank"]

    $("#ranking-content").fadeOut(function() {
      $(this).text("Top "+ rank)
    }).fadeIn();
  }

  fill_hotels(hotel_default_id);
  fill_reviews(data, hotel_default_id)
  fill_rank(data, hotel_default_id)

  $(".hotel_card").click(function(e){
    $(".hotel_card").removeClass("active")
    $(this).addClass("active")
    id = $(this).data("id")
    fill_reviews(data, id)
    fill_rank(data, id)
    e.preventDefault()
  });

})

