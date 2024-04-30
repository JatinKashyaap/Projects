SELECT
	time,
	(CASE
		WHEN `time` BETWEEN "00:00:00" AND "12:00:00" THEN "Morning"
        WHEN `time` BETWEEN "12:01:00" AND "16:00:00" THEN "Afternoon"
        ELSE "Evening"
    END) AS time_of_day
FROM sales;
ALTER TABLE sales ADD COLUMN time_of_day VARCHAR(20);
UPDATE sales
SET time_of_day = (
	CASE
		WHEN `time` BETWEEN "00:00:00" AND "12:00:00" THEN "Morning"
        WHEN `time` BETWEEN "12:01:00" AND "16:00:00" THEN "Afternoon"
        ELSE "Evening"
    END
);

-- Add day_name column
SELECT
	date,
	DAYNAME(date)
FROM sales;

ALTER TABLE sales ADD COLUMN day_name VARCHAR(10);

UPDATE sales
SET day_name = DAYNAME(date);
-- Add month_name column
SELECT
	date,
	MONTHNAME(date)
FROM sales;

ALTER TABLE sales ADD COLUMN month_name VARCHAR(10);

UPDATE sales
SET month_name = MONTHNAME(date);

-- Generic Questions
select* from sales;
-- 1. How many unique cities does the data have?
select distinct city from sales;

-- 2. In which city is the each branch?
select distinct city,branch from sales;
-- Product
-- 3. How many unique product lines does the data have?

select distinct product_line from sales;

-- 4. What is the most selling product line?

select sum(quantity) as qty,product_line
from sales
group by product_line
order by qty desc;

-- 5. What is the total revenue by month?
select month_name AS  month,
sum(total) as revenue 
from sales
group by month_name
order by revenue desc;
-- 6.which month has the largest COGS?
select month_name as month,
sum(cogs) as total_cogs
from sales
group by month_name
order by total_cogs desc;
-- 7.Which branch sold more products than average product sold ?
select branch,sum(quantity) as qty
from sales
group by branch
having sum(quantity) > (select avg (quantity) from sales);
-- 8.What is the most common product line by gender?
select gender,product_line,count(gender) as total_cnt
from sales
group by gender,product_line
order by total_cnt  desc;
-- 9. What is the average rating of each product line?
select avg(rating) as avg_rating,product_line from sales
group by product_line
order by avg_rating desc;
-- 10. What is the most common payment method?
select payment,count(payment) as cnt_payment from sales
group by payment
order by cnt_payment desc;
-- 11.Fetch each product line and add a column to those product line showing "Good", "Bad". Good if its greater than average sales
select product_line,
(case 
      when  avg(total) > sum(total) then "Good"
      else "bad"
      end) as remark 
      from sales 
      group by product_line;      
   ALTER TABLE sales ADD COLUMN remark VARCHAR(30);   
   UPDATE sales
SET remark = (
	CASE 
    when  avg(quantity) > sum(quantity) then "Good"
      else "bad"
      end);
-- 12. What product line had the largest VAT?

SELECT
	product_line,
	AVG(tax_pct) as avg_tax
FROM sales
GROUP BY product_line
ORDER BY avg_tax DESC;
-- Sales Questions--
-- 1. Number of sales made in each time of the day per weekday?
select time_of_day,count(*) as total_sales from sales
where day_name ="Sunday"
group by time_of_day
order by total_sales  desc;
-- 2 Which of the customer types brings the most revenue?
select customer_type, sum(total) as total_revenue from sales
group by customer_type
order by total_revenue desc;
 -- 3.Which city has the largest tax percent/ VAT (Value Added Tax)?
 select city, avg(tax_pct) as vat
 from sales
 group by city
 order by vat desc;
 -- 4. Which customer type pays the most in VAT?
 select customer_type , avg(tax_pct) as total_vat
 from sales
 group by customer_type
 order by total_vat desc;
 -- -------------------------------------------------------------------------------
 -- -------------------------Customer---------------------------------------------
 -- 1. How many unique customer types does the data have?
 select distinct(customer_type) from sales;
 
  -- 2.How many unique payment methods does the data have?
  select distinct(payment) from sales;
  -- 3. What is the most common customer type?
  select customer_type, count(customer_type) as cnt_customer from sales
  group by customer_type
  order by cnt_customer;
-- 4. Which customer type buys the most?
select customer_type, count( customer_type) as total_qty from sales
group by customer_type;
 -- 5.What is the gender of most of the customers?
 select gender, count(gender) as gender_count from sales
 group by gender;
 -- 6. What is the gender distribution per branch?
 select gender, count(gender) as gender_count from sales
 where branch="C"
 group by gender
 order by gender_count desc;
  -- 7.Which time of the day do customers give most ratings?
  select time_of_day, avg(rating) as avg_rating
  from sales
  group by time_of_day
  order by avg_rating desc;
  -- 8. Which time of the day do customers give most ratings per branch?
select time_of_day, avg(rating) as avg_rating
  from sales
  where branch="B"
  group by time_of_day
  order by avg_rating desc;
 -- 9.Which day fo the week has the best avg ratings?
 select day_name ,avg(rating) as avg_rating
  from sales
  group by day_name
  order by avg_rating desc; 
  -- 10.Which day of the week has the best average ratings per branch?
  select day_name ,avg(rating) as avg_rating
  from sales
  where branch="A"
  group by day_name
  order by avg_rating desc; 

	





