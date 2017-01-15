<!DOCTYPE html>
<!--[if lt IE 8 ]><html class="no-js ie ie7" lang="en"> <![endif]-->
<!--[if IE 8 ]><html class="no-js ie ie8" lang="en"> <![endif]-->
<!--[if (gte IE 8)|!(IE)]><!--><html class="no-js" lang="en"> <!--<![endif]-->
<head>

   <!--- Basic Page Needs
   ================================================== -->
   <meta charset="utf-8">
	<title>ASD Prediction</title>
	<meta name="description" content="Autism Prediction">
	<meta name="author" content="Harshit Kumar Gupta">

   <!-- Mobile Specific Metas
   ================================================== -->
	<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">

	<!-- CSS
    ================================================== -->
   <link rel="stylesheet" href="css/default.css">
	<link rel="stylesheet" href="css/layout.css">
   <link rel="stylesheet" href="css/media-queries.css">
   <link rel="stylesheet" href="css/magnific-popup.css">

   <!-- Script
   ================================================== -->
	<script src="js/modernizr.js"></script>

   <!-- Favicons
	================================================== -->
	<link rel="shortcut icon" href="favicon.png" >

</head>

<body>
<script>
  window.fbAsyncInit = function() {
    FB.init({
      appId      : '543134095832041',
      xfbml      : true,
      version    : 'v2.1'
    });
  };

  (function(d, s, id){
     var js, fjs = d.getElementsByTagName(s)[0];
     if (d.getElementById(id)) {return;}
     js = d.createElement(s); js.id = id;
     js.src = "//connect.facebook.net/en_US/sdk.js";
     fjs.parentNode.insertBefore(js, fjs);
   }(document, 'script', 'facebook-jssdk'));

function postLike() {
    FB.api(
       'https://graph.facebook.com/me/og.likes',
       'post',
       { object: objectToLike,
         privacy: {'value': 'SELF'} },
       function(response) {
         if (!response) {
           alert('Error occurred.');
         } else if (response.error) {
           document.getElementById('result').innerHTML =
             'Error: ' + response.error.message;
         } else {
           document.getElementById('result').innerHTML =
             '<a href=\"https://www.facebook.com/me/activity/' +
             response.id + '\">' +
             'Story created.  ID is ' +
             response.id + '</a>';
         }
       }
    );
  }
</script>
   <!-- Header
   ================================================== -->
   <header id="home">

      <nav id="nav-wrap">

         <a class="mobile-btn" href="#nav-wrap" title="Show navigation">Show navigation</a>
	      <a class="mobile-btn" href="#" title="Hide navigation">Hide navigation</a>

         <ul id="nav" class="nav">
            <li class="current"><a class="smoothscroll" href="#home">Home</a></li>
            <li><a class="smoothscroll" href="#about">About</a></li>
	      <!--   <li><a class="smoothscroll" href="#resume">Resume</a></li>
            <li><a class="smoothscroll" href="#portfolio">Works</a></li>
            <li><a class="smoothscroll" href="#testimonials">Testimonials</a></li>-->
		 <li><a class="smoothscroll" href="#resume">How to Use</a></li>
		 <li><a class="smoothscroll" href="#portfolio">Download</a></li>
            <li><a class="smoothscroll" href="#contact">Contact</a></li>
         </ul> <!-- end #nav -->

      </nav> <!-- end #nav-wrap -->

      <div class="row banner">
         <div class="banner-text">
            <h1 class="responsive-headline">Autism Spectrum Disorder Prediction</h1>
<div
  class="fb-like"
  data-share="true"
  data-width="450"
  data-show-faces="true">
</div>
<div
  class="fb-login-button"
  data-show-faces="true"
  data-width="200"
  data-max-rows="1"
  data-scope="publish_actions">
</div>

            <h3>It is project run in collaboration by NBRC and IIT DELHI. It is dedicated for predicting Autism in Toddlers. Download Application Application to your device.Record Speech sample of  your children and upload to server so that we can give you feedback of your child</h3>
            <hr />
            <ul class="social">
               <li><a href="https://www.facebook.com/nbrcindia"><i class="fa fa-facebook"></i></a></li>
               <li><a href="#"><i class="fa fa-twitter"></i></a></li>
               <li><a href="#"><i class="fa fa-google-plus"></i></a></li>
               <li><a href="#"><i class="fa fa-linkedin"></i></a></li>
               <li><a href="#"><i class="fa fa-instagram"></i></a></li>
               <li><a href="#"><i class="fa fa-dribbble"></i></a></li>
               <li><a href="#"><i class="fa fa-skype"></i></a></li>
            </ul>
         </div>
      </div>

      <p class="scrolldown">
         <a class="smoothscroll" href="#about"><i class="icon-down-circle"></i></a>
      </p>

   </header> <!-- Header End -->


   <!-- About Section
   ================================================== -->
   <section id="about">

      <div class="row">

         <div class="three columns">

            <img class="profile-pic"  src="images/nbrc.png" alt="" />



         </div>

         <div class="nine columns main-col">

            <h2>About Institute</h2>

            <p> National Brain Research Centre is the only institute in India dedicated to neuroscience research and education. Scientists and students of NBRC come from diverse academic backgrounds, including biological, computational, mathematical, physical, engineering and medical sciences, and use multidisciplinary approaches to understand the brain.

Located in the foothills of the Aravali range in Manesar, Haryana, NBRC is an autonomous institute funded by the Department of Biotechnology, Government of India, and is also a Deemed University. 
            </p>

          

          

         </div> <!-- end .main-col -->

      </div>

   </section> <!-- About Section End-->


   <!-- How to Use
   ================================================== -->
   <section id="resume">

    
      <div class="row education">

         <div class="three columns header-col">
            <h1><span>HOW TO USE</span></h1>
         </div>

         <div class="nine columns main-col">

            <div class="row item">

               <div class="twelve columns">

                  <h3>Allow installation of non market apps</h3>
                  <p class="info">mark Unknown Sources</p>

                  <p>
                  <img class="profile-pic"  src="images/1.png" alt="" />
	          		
		  <img class="profile-pic"  src="images/2.png" alt="" />
                  </p>

               </div>

            </div> <!-- item end -->

            <div class="row item">

               <div class="twelve columns">

                  <h3>Intallation of application</h3>
                  <p class="info">use Package Installer</p>

                  <p>
                  <img class="profile-pic"  src="images/3.png" alt="" />
	          		
		  <img class="profile-pic"  src="images/4.png" alt="" />
                  </p>

               </div>

            </div> <!-- item end -->
		<div class="row item">

               <div class="twelve columns">

                  <h3>Starting of Application</h3>
                  <p class="info">Turn on internet connection</p>

                  <p>
                  <img class="profile-pic"  src="images/5.png" alt="" />
	          		
		  <img class="profile-pic"  src="images/6.png" alt="" />
                  </p>

               </div>

            </div> <!-- item end -->
<div class="row item">

               <div class="twelve columns">

                  <h3>Viewing GCM messages</h3>
                  <p class="info">received messages send from gcm server</p>

                  <p>
                  <img class="profile-pic"  src="images/7.png" alt="" />
	          		
		  
                  </p>

               </div>

            </div> <!-- item end -->
<div class="row item">

               <div class="twelve columns">

                  <h3>Menu </h3>
                  <p class="info">menu button  or top right bar</p>

                  <p>
                  <img class="profile-pic"  src="images/8.png" alt="" />
	          		
		  
                  </p>

               </div>

            </div> <!-- item end -->
<div class="row item">

               <div class="twelve columns">

                  <h3>Recording of Speech Sample</h3>
                  <p class="info">recorded files saved in AudioRecorder Folder</p>

                  <p>
                  <img class="profile-pic"  src="images/9.png" alt="" />
	          		
		  <img class="profile-pic"  src="images/10.png" alt="" />
	<img class="profile-pic"  src="images/11.png" alt="" />
                  </p>

               </div>

            </div> <!-- item end -->
<div class="row item">

               <div class="twelve columns">

                  <h3>Uploading of Speech Sample</h3>
                  <p class="info">local selected file is shownr</p>

                  <p>
                  <img class="profile-pic"  src="images/12.png" alt="" />
	          		
		  <img class="profile-pic"  src="images/13.png" alt="" />
	<img class="profile-pic"  src="images/14.png" alt="" />
	<img class="profile-pic"  src="images/15.png" alt="" />
	<img class="profile-pic"  src="images/16.png" alt="" />
                  </p>

               </div>

            </div> <!-- item end -->
         </div> <!-- main-col end -->

      </div> <!-- End Education -->


 

         


    

   </section> <!-- Resume Section End-->
<section id="portfolio">

      <div class="row">

         <div class="two columns header-col">

            <h1>Download Android Application</h1>

         </div>

         <div class="seven columns">

           <FORM ACTION="download.php" METHOD="GET">


<INPUT TYPE="hidden" NAME="file"    VALUE= "GCM.apk"  >


 <INPUT TYPE="image" SRC="images/download.jpg" ALT="Submit"></FORM>
            
	     		
         </div>

        

      </div>

   </section> <!-- Call-To-Action Section End-->



   <!-- Contact Section
   ================================================== -->
   <section id="contact">

         <div class="row section-head">

            <div class="two columns header-col">

               <h1><span>Get In Touch.</span></h1>

            </div>

            <div class="ten columns">

                  <p class="lead">Email any Quries and Suggestion for improving our web and app interface.
                  </p>

            </div>

         </div>

         <div class="row">

            <div class="eight columns">

               <!-- form -->
               <form action="" method="post" id="contactForm" name="contactForm">
					<fieldset>

                  <div>
						   <label for="contactName">Name <span class="required">*</span></label>
						   <input type="text" value="" size="35" id="contactName" name="contactName">
                  </div>

                  <div>
						   <label for="contactEmail">Email <span class="required">*</span></label>
						   <input type="text" value="" size="35" id="contactEmail" name="contactEmail">
                  </div>

                  <div>
						   <label for="contactSubject">Subject</label>
						   <input type="text" value="" size="35" id="contactSubject" name="contactSubject">
                  </div>

                  <div>
                     <label for="contactMessage">Message <span class="required">*</span></label>
                     <textarea cols="50" rows="15" id="contactMessage" name="contactMessage"></textarea>
                  </div>

                  <div>
                     <button class="submit">Submit</button>
                     <span id="image-loader">
                        <img alt="" src="images/loader.gif">
                     </span>
                  </div>

					</fieldset>
				   </form> <!-- Form End -->

               <!-- contact-warning -->
               <div id="message-warning"> Error boy</div>
               <!-- contact-success -->
				   <div id="message-success">
                  <i class="fa fa-check"></i>Your message was sent, thank you!<br>
				   </div>

            </div>


            <aside class="four columns footer-widgets">

               <div class="widget widget_contact">

					   <h4>Address and Phone</h4>
					   <p class="address">
						  NH-8, Manesar, Gurgaon,<br>
							Haryana - 122 051, India<br>	
						   <span>Tel.:-   91-124 – 2845 200<br>
							Fax:- 	91-124 - 233 89 10 <br>
							91-124 - 233 89 28 <br>
							Email:- info@nbrc.ac.in</span>
					   </p>

				   </div>

               <div class="widget widget_tweets">

                  

		         </div>

            </aside>

      </div>

   </section> <!-- Contact Section End-->


   <!-- footer
   ================================================== -->
   <footer>

      <div class="row">

         <div class="twelve columns">

            <ul class="social-links">
               <li><a href="https://www.facebook.com/nbrcindia"><i class="fa fa-facebook"></i></a></li>
               <li><a href="#"><i class="fa fa-twitter"></i></a></li>
               <li><a href="#"><i class="fa fa-google-plus"></i></a></li>
               <li><a href="#"><i class="fa fa-linkedin"></i></a></li>
               <li><a href="#"><i class="fa fa-instagram"></i></a></li>
               <li><a href="#"><i class="fa fa-dribbble"></i></a></li>
               <li><a href="#"><i class="fa fa-skype"></i></a></li>
            </ul>

            <ul class="copyright">
               
            </ul>

         </div>

         <div id="go-top"><a class="smoothscroll" title="Back to Top" href="#home"><i class="icon-up-open"></i></a></div>

      </div>

   </footer> <!-- Footer End-->

   <!-- Java Script
   ================================================== -->
   <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>
   <script>window.jQuery || document.write('<script src="js/jquery-1.10.2.min.js"><\/script>')</script>
   <script type="text/javascript" src="js/jquery-migrate-1.2.1.min.js"></script>

   <script src="js/jquery.flexslider.js"></script>
   <script src="js/waypoints.js"></script>
   <script src="js/jquery.fittext.js"></script>
   <script src="js/magnific-popup.js"></script>
   <script src="js/init.js"></script>

</body>

</html>