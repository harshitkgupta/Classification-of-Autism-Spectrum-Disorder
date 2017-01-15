<?php 
 require_once('loader.php');
$dir="../uploads";
$output="";
$email=$_POST["email"];
$gcmRegID= $_POST["regId"]; 

if(($_FILES["file"]["size"]<20000000))
{
	if($_FILES["file"]["error"]>0)
	{
		$output=$output."Error ".$_FILES["file"]["error"]."</br>";
	}
	else
	{
		$output=$output."Upload ".$_FILES["file"]["name"]."<br/>";
		$output=$output."type ".$_FILES["file"]["type"]."<br/>";
		$output=$output."size ".$_FILES["file"]["size"]."<br/>";
		$output=$output."initially Stored in ".$_FILES["file"]["tmp_name"]."<br/>";
		
		$target_path = $dir.'/'.$email."/";
		if(!is_dir($dir)) 
		{
			    mkdir($dir);         
		}
		$target_path =	$target_path.basename( $_FILES['file']['name']);
		$output=$output."target path     ".$target_path."<br/>";
		if(file_exists($target_path))
		{
			$output=$output.$_FILES["file"]["name"]."already exists";
		}
		else
		{
			//if(move_uploaded_file($_FILES["file"]["tmp_name"],$target_path))
			{
				$output=$output."finally data Stored in ".$target_path;
				//chmod ($target_path, 0644);
			}			
		//	else
				$output=$output."file can not be moved in upload directory";
		}
	}
}
else
	$output=$output." file of very big size";
echo $output
/*
if (isset($gcmRegID) && isset($output)) {
	
		
		$registatoin_ids = array($gcmRegID);
		$message = array("price" => $output);
	
		$result = send_push_notification($registatoin_ids, $message);
	
		echo $result;
	}*/
?>