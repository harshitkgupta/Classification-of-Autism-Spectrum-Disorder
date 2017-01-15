<?
header('Content-Type: text/html; charset=utf-8');
$host = $_SERVER['HTTP_HOST'];
setlocale(LC_TIME, "in_IN.utf8");
date_default_timezone_set('Asia/Kolkata');

/*
Directory Listing Script - Version 2
====================================
Script Author: Ash Young <ash@evoluted.net>. www.evoluted.net
Layout: Manny <manny@tenka.co.uk>. www.tenka.co.uk
*/
$startdir = '.';
$showthumbnails = false; 
$showdirs = true;
$forcedownloads = false;
$hide = array(
				'dlf',
				'public_html',				
				'index.php',
				'Thumbs',
				'.htaccess',
				'.htpasswd'
			);
$displayindex = false;
$allowuploads = false;
$overwrite = false;

$indexfiles = array (
				'index.html',
				'index.htm',
				'default.htm',
				'default.html'
			);
			
$filetypes = array (
				'png' => 'jpg.gif',
				'jpeg' => 'jpg.gif',
				'bmp' => 'jpg.gif',
				'jpg' => 'jpg.gif', 
				'gif' => 'gif.gif',
				'zip' => 'archive.png',
				'rar' => 'archive.png',
				'exe' => 'exe.gif',
				'setup' => 'setup.gif',
				'txt' => 'text.png',
				'htm' => 'html.gif',
				'html' => 'html.gif',
				'php' => 'php.gif',				
				'fla' => 'fla.gif',
				'swf' => 'swf.gif',
				'xls' => 'xls.gif',
				'doc' => 'doc.gif',
				'sig' => 'sig.gif',
				'fh10' => 'fh10.gif',
				'pdf' => 'pdf.gif',
				'psd' => 'psd.gif',
				'rm' => 'real.gif',
				'mpg' => 'video.gif',
				'mpeg' => 'video.gif',
				'mov' => 'video2.gif',
				'avi' => 'video.gif',
				'eps' => 'eps.gif',
				'gz' => 'archive.png',
				'asc' => 'sig.gif',
			);
			
error_reporting(0);
if(!function_exists('imagecreatetruecolor')) $showthumbnails = false;
$leadon = $startdir;
if($leadon=='.') $leadon = '';
if((substr($leadon, -1, 1)!='/') && $leadon!='') $leadon = $leadon . '/';
$startdir = $leadon;

if($_GET['dir']) {
	// check this is okay.
	
	if(substr($_GET['dir'], -1, 1)!='/') {
		$_GET['dir'] = $_GET['dir'] . '/';
	}
	
	$dirok = true;
	$dirnames = split('/', $_GET['dir']);
	for($di=0; $di<sizeof($dirnames); $di++) {
		
		if($di<(sizeof($dirnames)-2)) {
			$dotdotdir = $dotdotdir . $dirnames[$di] . '/';
		}
		
		if($dirnames[$di] == '..') {
			$dirok = false;
		}
	}
	
	if(substr($_GET['dir'], 0, 1)=='/') {
		$dirok = false;
	}
	
	if($dirok) {
		 $leadon = $leadon . $_GET['dir'];
	}
}



$opendir = $leadon;
if(!$leadon) $opendir = '.';
if(!file_exists($opendir)) {
	$opendir = '.';
	$leadon = $startdir;
}

clearstatcache();
if ($handle = opendir($opendir)) {
	while (false !== ($file = readdir($handle))) { 
		// first see if this file is required in the listing
		if ($file == "." || $file == "..")  continue;
		$discard = false;
		for($hi=0;$hi<sizeof($hide);$hi++) {
			if(strpos($file, $hide[$hi])!==false) {
				$discard = true;
			}
		}
		
		if($discard) continue;
		if (@filetype($leadon.$file) == "dir") {
			if(!$showdirs) continue;
		
			$n++;
			if($_GET['sort']=="date") {
				$key = @filemtime($leadon.$file) . ".$n";
			}
			else {
				$key = $n;
			}
			$dirs[$key] = $file . "/";
		}
		else {
			$n++;
			if($_GET['sort']=="date") {
				$key = @filemtime($leadon.$file) . ".$n";
			}
			elseif($_GET['sort']=="size") {
				$key = @filesize($leadon.$file) . ".$n";
			}
			else {
				$key = $n;
			}
			$files[$key] = $file;
			
			if($displayindex) {
				if(in_array(strtolower($file), $indexfiles)) {
					header("Location: $file");
					die();
				}
			}
		}
	}
	closedir($handle); 
}

// sort our files
if($_GET['sort']=="date") {
	@ksort($dirs, SORT_NUMERIC);
	@ksort($files, SORT_NUMERIC);
}
elseif($_GET['sort']=="size") {
	@natcasesort($dirs); 
	@ksort($files, SORT_NUMERIC);
}
else {
	@natcasesort($dirs); 
	@natcasesort($files);
}

// order correctly
if($_GET['order']=="desc" && $_GET['sort']!="size") {$dirs = @array_reverse($dirs);}
if($_GET['order']=="desc") {$files = @array_reverse($files);}
$dirs = @array_values($dirs); $files = @array_values($files);

?>
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
    <head>
        <title>Welcome to <? print $host; ?>!</title>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <link href="http://www.main-hosting.com/hostinger/welcome/css/site.css" media="screen" rel="stylesheet" type="text/css" />
    </head>
    <body>
        <div id="main">
            <div id="content">
                <div class="header">
                    <a id="logo" href="http://www.hostinger.in/"><img src="http://www.main-hosting.com/hostinger/welcome/images/logo.png" alt="Web hosting" /></a>
                </div>
                <div class="content">
                    <h1>Your account has been created!</h1>
                    <p>Website <b><? print $host; ?></b> has been successfully installed on the server! Please delete the file <b>default.php</b> from the <b>public_html</b> folder and then upload your website by using FTP or File Manager.</p>
                    <p>Here is a list of files in your public_html folder:</p>
                    <div id="files">
                        <div class="top"></div>
                        <div class="cont">

                            <div id="listingcontainer">
                                <div id="listing">
                                <?
                                $class = 'b';
                                if($dirok) {
                                ?>
                                  <div><a href="<?=$dotdotdir;?>" class="<?=$class;?>"><img src="http://www.main-hosting.com/hostinger/welcome/index/dirup.png" alt="Folder" /><strong>..</strong> <em>-</em><? $mtime = filemtime($dotdotdir); $mtime = date("m/d/Y H:i:s", $mtime); $mtime = strftime("%B %e, %G %T", strtotime($mtime)); print ucfirst($mtime); ?></a></div>
                                <?
                                    if($class=='b') $class='w';
                                    else $class = 'b';
                                }
                                $arsize = sizeof($dirs);
                                for($i=0;$i<$arsize;$i++) {
                                ?>
                                  <div><a href="<?=$leadon.$dirs[$i];?>" class="<?=$class;?>"><img src="http://www.main-hosting.com/hostinger/welcome/index/folder.png" alt="<?=$dirs[$i];?>" /><strong><?=$dirs[$i];?></strong> <em>-</em><? $mtime = filemtime($leadon.$dirs[$i]); $mtime = date("m/d/Y H:i:s", $mtime); $mtime = strftime("%B %e, %G %T", strtotime($mtime)); print ucfirst($mtime); ?></a></div>
                                <?
                                    if($class=='b') $class='w';
                                    else $class = 'b';	
                                }
                                
                                $arsize = sizeof($files);
                                for($i=0;$i<$arsize;$i++) {
                                    $icon = 'unknown.png';
                                    $ext = strtolower(substr($files[$i], strrpos($files[$i], '.')+1));
                                    $supportedimages = array('gif', 'png', 'jpeg', 'jpg');
                                    $thumb = '';
                                            
                                    if($filetypes[$ext]) {
                                        $icon = $filetypes[$ext];
                                    }
                                    
                                    $filename = $files[$i];
                                    if(strlen($filename)>43) {
                                        $filename = substr($files[$i], 0, 40) . '...';
                                    }
                                    
                                    $fileurl = $leadon . $files[$i];
                                ?>
                                  <div><a href="<?=$fileurl;?>" class="<?=$class;?>"<?=$thumb2;?>><img src="http://cpanel.main-hosting.com/images/index/<?=$icon;?>" alt="<?=$files[$i];?>" /><strong><?=$filename;?></strong><em><?=round(filesize($leadon.$files[$i])/1024);?> KB</em><? $mtime = filemtime($leadon.$files[$i]); $mtime = date("m/d/Y H:i:s", $mtime); $mtime = strftime("%B %e, %G %T", strtotime($mtime)); print ucfirst($mtime); ?><?=$thumb;?></a></div>
                                <?
                                    if($class=='b') $class='w';
                                    else $class = 'b';	
                                }	
                                ?>
                                </div>
                            </div>

                        </div>
                        <div class="bottom"></div>
                        <div class="clear"></div>
                    </div>
                    <div class="clear"></div>
                </div>
                <div class="footer"></div>
                <div class="clear"></div>
            </div>
            <div id="footer">
                <div class="links">
                    <a href="http://www.hostinger.in/web-hosting" target="_blank">Web Hosting</a> 
                    <span class="pipe">|</span> 
                    <a href="http://www.hostinger.in/free-hosting" target="_blank">Free Hosting</a> 
                    <span class="pipe">|</span> 
                    <a href="http://www.hostinger.in/forum" target="_blank">Support Forum</a> 
                    <span class="pipe">|</span> 
                    <a href="http://cpanel.hostinger.in/" target="_blank">Client Login</a>
                </div>
                <div class="copyright">Hostinger India &copy; <? print date('Y'); ?>. All rights reserved</div>
                <div class="social-icons">
                    <a href="https://www.facebook.com/pages/Hostinger-India/515296155192316"><img src="http://www.main-hosting.com/hostinger/welcome/images/fb.gif" /></a>
                    <a href="https://twitter.com/HostingerCOM"><img src="http://www.main-hosting.com/hostinger/welcome/images/twitter.gif" /></a>
                </div>
            </div>
        </div>
    </body>
</html>